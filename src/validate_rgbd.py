"""离线检查一组 RGB-D 的深度单位、焦点与直边配准。"""

import argparse
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np


def load_sample(folder):
    color = cv2.imdecode(np.frombuffer((folder / "rgb.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("无法解码 rgb.png。")
    raw = np.load(folder / "depth_raw.npy", allow_pickle=False)
    depth = np.load(folder / "depth_m.npy", allow_pickle=False)
    meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("meta.json 应包含一个 JSON 对象。")
    if raw.ndim != 2 or raw.dtype != np.uint16 or not raw.size:
        raise ValueError("depth_raw.npy 应为非空的二维 uint16 数组。")
    if depth.shape != color.shape[:2] or depth.dtype.kind != "f":
        raise ValueError("depth_m.npy 应为与彩色图等高宽的二维浮点数组。")
    if np.isinf(depth).any() or np.any(depth <= 0) or np.isnan(depth).all():
        raise ValueError("对齐深度应含有限正数，无效值应为 NaN。")
    if meta.get("depth_m_unit") != "m" or meta.get("depth_m_invalid") != "NaN":
        raise ValueError("meta.json 应声明对齐深度单位为 m，缺测为 NaN。")
    if meta.get("rgb_size_wh") != list(color.shape[:2][::-1]):
        raise ValueError("meta.json 的 rgb_size_wh 与 rgb.png 不符。")
    if meta.get("raw_depth_size_wh") != list(raw.shape[::-1]):
        raise ValueError("meta.json 的 raw_depth_size_wh 与 depth_raw.npy 不符。")
    for field in ("raw_depth_mm_per_unit", "aligned_depth_mm_per_unit_before_conversion"):
        scale = meta.get(field)
        if type(scale) not in (int, float) or not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"meta.json 缺少有效深度比例 {field}。")
    for field in ("color_timestamp_us", "depth_timestamp_us"):
        timestamp = meta.get(field)
        if type(timestamp) is not int or timestamp < 0:
            raise ValueError(f"meta.json 缺少有效帧时间戳 {field}。")
    sdk_version = meta.get("sdk_package_version")
    if not isinstance(sdk_version, str) or not sdk_version.strip():
        raise ValueError("meta.json 缺少 SDK 版本。")
    return color, depth, meta


def check_points(points, shape):
    height, width = shape
    if np.any(points < 0) or np.any(points >= [width, height]):
        raise ValueError(f"坐标超出彩色图范围（宽 {width}，高 {height}）。")


def read_focus(depth, xy, measured_m):
    x, y = xy
    center = float(depth[y, x])
    if not np.isfinite(center):
        raise ValueError(f"焦点 ({x}, {y}) 位于深度空洞，请改选物体内部。")
    neighborhood = depth[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
    median = float(np.nanmedian(neighborhood))
    return {
        "pixel_depth_m": center,
        "median_5x5_m": median,
        "measured_m": measured_m,
        "median_minus_measured_m": median - measured_m,
    }


def pixel(xy):
    return tuple(np.rint(xy).astype(int).tolist())


def measure_edge(color, depth, line, foreground, background, radius, min_jump_m):
    start, end = line
    tangent = end - start
    length = np.linalg.norm(tangent)
    if length < 14:
        raise ValueError("直边线段请至少标出 14 像素，并避开拐角。")
    normal = np.array([-tangent[1], tangent[0]]) / length
    if (background - start) @ normal < 0:
        normal = -normal
    if (foreground - start) @ normal >= 0 or (background - start) @ normal <= 0:
        raise ValueError("前景和背景焦点应位于标注直边的两侧。")

    # 沿线等距取七个截面。两幅图在相同像素取样，深度空洞不参与差分。
    offsets = np.arange(-radius, radius + 1)
    mid_offsets = offsets[:-1] + 0.5
    near_line = np.flatnonzero(np.abs(mid_offsets) <= 3)
    overlay = color.copy()
    scans = []
    for fraction in np.linspace(0.1, 0.9, 7):
        center = start + fraction * tangent
        positions = center + offsets[:, None] * normal
        xy = np.rint(positions).astype(int)
        check_points(xy, depth.shape)
        x, y = xy.T
        colors = color[y, x].astype(np.float32)
        rgb_change = np.linalg.norm(np.diff(colors, axis=0), axis=1)
        rgb_index = near_line[np.argmax(rgb_change[near_line])]
        depths = depth[y, x]
        depth_change = np.diff(depths)
        missing = xy[np.isnan(depths)]
        scan = {
            "center_xy": center.round(2).tolist(),
            "rgb_edge_xy": None,
            "depth_edge_xy": None,
            "offset_px": None,
            "missing_depth_xy": np.unique(missing, axis=0).tolist(),
            "status": "no_rgb_edge",
        }
        cv2.line(overlay, pixel(xy[0]), pixel(xy[-1]), (255, 255, 255), 1)
        if rgb_change[rgb_index] >= 15:
            rgb_xy = center + mid_offsets[rgb_index] * normal
            scan["rgb_edge_xy"] = rgb_xy.round(2).tolist()
            scan["status"] = "no_depth_jump"
            cv2.circle(overlay, pixel(rgb_xy), 5, (0, 255, 0), 1)
            candidates = np.flatnonzero(depth_change >= min_jump_m)
            if candidates.size:
                depth_index = min(
                    candidates, key=lambda i: (-depth_change[i], abs(i - rgb_index))
                )
                depth_xy = center + mid_offsets[depth_index] * normal
                scan["depth_edge_xy"] = depth_xy.round(2).tolist()
                scan["offset_px"] = float(mid_offsets[depth_index] - mid_offsets[rgb_index])
                scan["status"] = "matched"
                cv2.circle(overlay, pixel(depth_xy), 2, (255, 0, 255), -1)
        for point in scan["missing_depth_xy"]:
            cv2.drawMarker(overlay, tuple(point), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 5)
        scans.append(scan)

    for label, point in (("F", foreground), ("B", background)):
        cv2.drawMarker(overlay, pixel(point), (0, 255, 255), cv2.MARKER_CROSS, 11)
        cv2.putText(overlay, label, pixel(point + [5, -5]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return scans, overlay


def validate_sample(args):
    if not args.scene_id.strip() or not args.config_version.strip():
        raise ValueError("场景编号和配置版本不能为空。")
    measured_front, measured_back = args.measured_m
    if not all(np.isfinite(args.measured_m)) or not 0 < measured_front < measured_back:
        raise ValueError("实测距离请用米填写，按从近到远的顺序输入两个正数。")
    color, depth, meta = load_sample(args.sample)
    foreground, background = np.array([args.foreground, args.background])
    line = np.array(args.edge).reshape(2, 2)
    check_points(np.vstack([foreground, background, line]), depth.shape)
    front = read_focus(depth, foreground, measured_front)
    back = read_focus(depth, background, measured_back)
    min_jump_m = 0.25 * (measured_back - measured_front)
    scans, overlay = measure_edge(
        color, depth, line, foreground, background, args.radius, min_jump_m
    )
    offsets = [scan["offset_px"] for scan in scans if scan["status"] == "matched"]
    order_ok = (front["pixel_depth_m"] < back["pixel_depth_m"]
                and front["median_5x5_m"] < back["median_5x5_m"])
    fraction = float(np.isfinite(depth).mean())
    meta.update(
        scene_id=args.scene_id.strip(),
        config_version=args.config_version.strip(),
        focus_points_xy={"foreground": foreground.tolist(), "background": background.tolist()},
        week1_validation={
            "foreground": front,
            "background": back,
            "foreground_is_nearer": order_ok,
            "valid_depth_fraction": fraction,
            "edge_line_xy": line.tolist(),
            "scan_radius_px": args.radius,
            "min_depth_jump_m": min_jump_m,
            "matched_scan_count": len(offsets),
            "scans_with_missing_depth": sum(bool(scan["missing_depth_xy"]) for scan in scans),
            "median_signed_offset_px": float(np.median(offsets)) if offsets else None,
            "median_abs_offset_px": float(np.median(np.abs(offsets))) if offsets else None,
            "max_abs_offset_px": float(np.max(np.abs(offsets))) if offsets else None,
            "scans": scans,
        },
    )
    success, png = cv2.imencode(".png", overlay)
    if not success:
        raise RuntimeError("无法编码配准标注图。")
    with TemporaryDirectory(dir=args.sample, prefix=".checking-") as temp:
        staged_meta = Path(temp) / "meta.json"
        staged_image = Path(temp) / "edge_check.png"
        staged_meta.write_text(
            json.dumps(meta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        staged_image.write_bytes(png.tobytes())
        staged_image.replace(args.sample / staged_image.name)
        staged_meta.replace(args.sample / staged_meta.name)

    print(f"前景 {front['median_5x5_m']:.3f} m（实测 {measured_front:.3f} m），"
          f"背景 {back['median_5x5_m']:.3f} m（实测 {measured_back:.3f} m）")
    if not order_ok:
        print("前景读数未小于背景，请核对焦点或深度数据。")
    print(f"有效深度 {fraction:.1%}，直边匹配 {len(offsets)}/{len(scans)} 条扫描线。")
    if offsets:
        print(f"有符号偏差中位数 {np.median(offsets):+.1f} px，正值表示深度边偏向背景。")
    print(f"已写入 {args.sample / 'meta.json'} 和 {args.sample / 'edge_check.png'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample", type=Path, help="一次采集生成的目录")
    parser.add_argument("--scene-id", required=True, help="场景编号")
    parser.add_argument("--config-version", default="week1-v1", help="采集配置版本")
    parser.add_argument("--foreground", nargs=2, type=int, required=True, metavar=("X", "Y"))
    parser.add_argument("--background", nargs=2, type=int, required=True, metavar=("X", "Y"))
    parser.add_argument("--measured-m", nargs=2, type=float, required=True, metavar=("NEAR", "FAR"))
    parser.add_argument("--edge", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--radius", type=int, default=12, help="每条直边扫描线的半长（像素）")
    args = parser.parse_args()
    if not 4 <= args.radius <= 100:
        parser.error("--radius 应在 4 至 100 像素之间。")
    try:
        validate_sample(args)
    except (ValueError, OSError, EOFError, RuntimeError, cv2.error) as error:
        print(f"检查失败：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
