"""采集并保存一组 Gemini 335 的彩色帧和深度帧。"""

import argparse
from datetime import datetime
from importlib.metadata import version
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import time

import cv2
import numpy as np
import pyorbbecsdk as ob


def color_to_bgr(frame):
    """把相机彩色帧转换成 OpenCV 使用的 BGR 图像。"""
    pixels = np.frombuffer(frame.get_data(), dtype=np.uint8)
    height, width = frame.get_height(), frame.get_width()
    pixel_format = frame.get_format()

    if pixel_format == ob.OBFormat.MJPG:
        image = cv2.imdecode(pixels, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError("彩色 MJPEG 帧解码失败。")
        return image
    if pixel_format == ob.OBFormat.BGR:
        return pixels.reshape(height, width, 3)

    conversions = {
        ob.OBFormat.RGB: (3, cv2.COLOR_RGB2BGR),
        ob.OBFormat.YUYV: (2, cv2.COLOR_YUV2BGR_YUY2),
        ob.OBFormat.UYVY: (2, cv2.COLOR_YUV2BGR_UYVY),
    }
    if pixel_format not in conversions:
        raise RuntimeError(f"暂未处理的彩色格式：{pixel_format}")
    channels, conversion = conversions[pixel_format]
    return cv2.cvtColor(pixels.reshape(height, width, channels), conversion)


def read_depth(frame):
    """从 Y16 帧读出原始整数和毫米比例，不修改相机缓冲区。"""
    if frame.get_format() != ob.OBFormat.Y16:
        raise RuntimeError(f"预期 Y16 深度帧，实际格式：{frame.get_format()}")
    shape = (frame.get_height(), frame.get_width())
    values = np.frombuffer(frame.get_data(), dtype=np.uint16).reshape(shape)
    scale = frame.get_depth_scale()
    if not np.isfinite(scale) or scale <= 0:
        raise RuntimeError(f"相机返回的深度比例无效：{scale}")
    return values, scale


def save_sample(output_root, color, raw_depth, aligned_depth_m, metadata):
    """整组文件写完后再生成最终目录，避免留下半组可见样本。"""
    output_root.mkdir(parents=True, exist_ok=True)
    folder = output_root / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    with TemporaryDirectory(dir=output_root, prefix=".writing-") as temporary:
        staging = Path(temporary) / "sample"
        staging.mkdir()

        # imencode 加 Path.write_bytes 可兼容 Windows 中文路径。
        success, png = cv2.imencode(".png", color)
        if not success:
            raise RuntimeError("无法编码彩色 PNG。")
        (staging / "rgb.png").write_bytes(png.tobytes())
        np.save(staging / "depth_raw.npy", raw_depth, allow_pickle=False)
        np.save(staging / "depth_m.npy", aligned_depth_m, allow_pickle=False)
        (staging / "meta.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging.rename(folder)
    return folder


def frame_exposure(frame):
    fields = ob.OBFrameMetadataType
    return {
        name: frame.get_metadata_value(field) if frame.has_metadata(field) else None
        for name, field in (
            ("auto_exposure", fields.AUTO_EXPOSURE),
            ("exposure_sdk_value", fields.EXPOSURE),
            ("gain_sdk_value", fields.GAIN),
        )
    }


def capture_one(output_root, lock_color_exposure=False):
    # 查询设备，避免误用电脑内置摄像头或其他 Orbbec 设备。
    context = ob.Context()
    devices = context.query_devices()
    for index in range(devices.get_count()):
        name = devices.get_device_name_by_index(index)
        if "gemini 335" in name.lower():
            break
    else:
        raise RuntimeError(
            "Orbbec SDK 未发现 Gemini 335。请在本机 PowerShell 运行，"
            "并确认 USB 数据线已连接、Orbbec Viewer 已关闭。"
        )

    device = devices.get_device_by_index(index)
    pipeline = ob.Pipeline(device)
    config = ob.Config()
    for sensor in (ob.OBSensorType.COLOR_SENSOR, ob.OBSensorType.DEPTH_SENSOR):
        profiles = pipeline.get_stream_profile_list(sensor)
        config.enable_stream(profiles.get_default_video_stream_profile())
    config.set_frame_aggregate_output_mode(ob.OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

    # 同步是让两路帧在时间上配对。对齐是把深度投影到彩色图坐标。
    pipeline.enable_frame_sync()
    color_ae = ob.OBPropertyID.OB_PROP_COLOR_AUTO_EXPOSURE_BOOL
    restore_color_ae = False
    locked_color_settings = None
    pipeline.start(config)
    try:
        align_to_color = ob.AlignFilter(ob.OBStreamType.COLOR_STREAM)
        align_to_color.set_match_target_resolution(True)

        deadline = time.monotonic() + 12
        warmup_pairs = 10
        print(f"已打开 {name}，正在等待彩色和深度帧……", flush=True)
        while time.monotonic() < deadline:
            frames = pipeline.wait_for_frames(1000)
            if frames is None:
                continue
            color_frame = frames.get_color_frame()
            raw_frame = frames.get_depth_frame()
            if color_frame is None or raw_frame is None:
                continue
            if warmup_pairs > 0:
                warmup_pairs -= 1
                continue
            if lock_color_exposure and locked_color_settings is None:
                exposure = ob.OBPropertyID.OB_PROP_COLOR_EXPOSURE_INT
                gain = ob.OBPropertyID.OB_PROP_COLOR_GAIN_INT
                exposure_value = device.get_int_property(exposure)
                gain_value = device.get_int_property(gain)
                if device.get_bool_property(color_ae):
                    restore_color_ae = True
                    device.set_bool_property(color_ae, False)
                    device.set_int_property(exposure, exposure_value)
                    device.set_int_property(gain, gain_value)
                if device.get_bool_property(color_ae):
                    raise RuntimeError("无法锁定彩色曝光，请检查相机的曝光设置。")
                locked_color_settings = {
                    "exposure_sdk_value": device.get_int_property(exposure),
                    "gain_sdk_value": device.get_int_property(gain),
                }
                warmup_pairs = 10
                continue

            # 保存对齐前的深度。复制后即使 SDK 释放帧，数组仍可使用。
            raw_depth, raw_scale = read_depth(raw_frame)
            raw_depth = raw_depth.copy()
            color = color_to_bgr(color_frame)
            color_timestamp = color_frame.get_timestamp_us()
            depth_timestamp = raw_frame.get_timestamp_us()
            color_exposure = frame_exposure(color_frame)
            depth_exposure = frame_exposure(raw_frame)

            aligned = align_to_color.process(frames)
            if aligned is None:
                continue
            aligned_frames = aligned.as_frame_set()
            if aligned_frames is None:
                continue
            aligned_frame = aligned_frames.get_depth_frame()
            if aligned_frame is None:
                continue
            aligned_values, aligned_scale = read_depth(aligned_frame)
            if aligned_values.shape != color.shape[:2]:
                raise RuntimeError("对齐后的深度尺寸与彩色图不一致。")

            # 原始值 × SDK 比例得到毫米，再除以 1000 转成米。
            depth_m = aligned_values.astype(np.float32) * (aligned_scale / 1000.0)
            valid = aligned_values != 0
            if not valid.any():
                continue
            depth_m[~valid] = np.nan
            metadata = {
                "camera": name,
                "sdk_package_version": version("pyorbbecsdk2"),
                "captured_at": datetime.now().astimezone().isoformat(),
                "color_format": str(color_frame.get_format()),
                "color_timestamp_us": color_timestamp,
                "depth_timestamp_us": depth_timestamp,
                "locked_color_settings": locked_color_settings,
                "color_frame_exposure": color_exposure,
                "depth_frame_exposure": depth_exposure,
                "rgb_size_wh": [color.shape[1], color.shape[0]],
                "raw_depth_size_wh": [raw_depth.shape[1], raw_depth.shape[0]],
                "raw_depth_mm_per_unit": raw_scale,
                "aligned_depth_mm_per_unit_before_conversion": aligned_scale,
                "depth_m_unit": "m",
                "depth_m_invalid": "NaN",
                "valid_depth_fraction": float(valid.mean()),
                "median_valid_depth_m": float(np.median(depth_m[valid])),
            }
            folder = save_sample(output_root, color, raw_depth, depth_m, metadata)
            print(f"已保存：{folder.resolve()}")
            print(
                f"有效深度占比：{metadata['valid_depth_fraction']:.1%}，"
                f"有效深度中位数：{metadata['median_valid_depth_m']:.3f} m"
            )
            return
        raise RuntimeError("12 秒内没有收到可保存的对齐帧，请检查相机连接和镜头前的场景。")
    finally:
        try:
            if restore_color_ae:
                device.set_bool_property(color_ae, True)
        finally:
            pipeline.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default_output = Path(__file__).resolve().parents[1] / "data" / "raw"
    parser.add_argument("--output", type=Path, default=default_output, help="采集结果父目录")
    parser.add_argument(
        "--lock-color-exposure", action="store_true", help="预热后锁定彩色曝光与增益，退出时恢复自动曝光"
    )
    args = parser.parse_args()
    try:
        capture_one(args.output, args.lock_color_exposure)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError, OSError, ob.OBError, cv2.error) as error:
        print(f"采集失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
