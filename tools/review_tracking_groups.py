# -*- coding: utf-8 -*-
"""Build original-frame/crop evidence for grouped tracking review; no speed truth."""
import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.shot_physics import select_flight

LABELS = {
    'real_primary': '실촬영 기본 후보 16개 — 촬영 조건 미확인',
    'zoomed_out': '실촬영 줌아웃 4개',
    'slow_motion': '테스트 슬로모션 4개 — 실제 배율 미확인',
    'possibly_different_camera_or_zoom': '테스트 카메라 설정 차이 의심 3개',
    'other_test': '나머지 테스트 5개 — 이동·관측 조건 별도 확인',
}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def font(size):
    return ImageFont.truetype('C:/Windows/Fonts/malgun.ttf', size)


def tile(frame, row, trusted, last_center):
    canvas = Image.new('RGB', (320, 420), '#111827')
    draw = ImageDraw.Draw(canvas)
    center = row.get('raw_center') or ([row['x'], row['y']] if row['x'] is not None else last_center)
    source = row['source']
    title = f"f{row['frame_number']} {source} | {'3D 선택' if trusted else '3D 미선택'}"
    draw.text((5, 3), title, font=font(15), fill='#67e8f9' if trusted else 'white')
    overview = frame.copy()
    if center is not None:
        point = tuple(round(p) for p in center)
        cv2.circle(overview, point, 24, (0, 255, 255), 3)
    canvas.paste(Image.fromarray(cv2.cvtColor(cv2.resize(overview, (320, 180)), cv2.COLOR_BGR2RGB)), (0, 28))
    draw.text((8, 211), '원본 확대', font=font(15), fill='white')
    draw.text((168, 211), '검출 박스·중심', font=font(15), fill='white')
    if center is not None:
        bbox = row.get('ball_bbox')
        side = round(max(64, min(256, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * 2.5 + 16))) if bbox else 96
        x0, y0 = round(center[0] - side / 2), round(center[1] - side / 2)
        crop = np.full((side, side, 3), 30, dtype=np.uint8)
        left, top, right, bottom = max(x0, 0), max(y0, 0), min(x0 + side, frame.shape[1]), min(y0 + side, frame.shape[0])
        if right > left and bottom > top:
            crop[top - y0:bottom - y0, left - x0:right - x0] = frame[top:bottom, left:right]
        overlay = crop.copy()
        color = (0, 220, 0) if source == 'yolo' else (0, 180, 255)
        if bbox is not None:
            cv2.rectangle(overlay, (round(bbox[0] - x0), round(bbox[1] - y0)), (round(bbox[2] - x0), round(bbox[3] - y0)), color, 1)
        cv2.drawMarker(overlay, (round(center[0] - x0), round(center[1] - y0)), (255, 0, 255), cv2.MARKER_CROSS, 7, 1)
        for index, source_image in enumerate([crop, overlay]):
            image = cv2.resize(source_image, (160, 160), interpolation=cv2.INTER_NEAREST)
            canvas.paste(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), (index * 160, 235))
        note = f"ROI=({x0},{y0}) {side}px | conf={row.get('confidence')}"
    else:
        note = '좌표 없음 / no crop position'
    draw.text((5, 398), note[:47], font=font(12), fill='#d1d5db')
    return canvas


def panel(cap, frames, numbers, selected, target, title, subtitle):
    height = 80 + 420 * math.ceil(len(numbers) / 4)
    canvas = Image.new('RGB', (1280, height), '#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 8), title, font=font(24), fill='white')
    draw.text((12, 42), subtitle, font=font(17), fill='#d1d5db')
    for index, number in enumerate(numbers):
        cap.set(cv2.CAP_PROP_POS_FRAMES, number)
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode frame {number}')
        earlier = next((r for r in reversed(frames[:number + 1]) if r['x'] is not None), None)
        last_center = [earlier['x'], earlier['y']] if earlier else None
        canvas.paste(tile(frame, frames[number], number in selected, last_center), ((index % 4) * 320, 80 + (index // 4) * 420))
    target.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(target, quality=92)
    return height


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'output/ai_baseline/stage3_20261010_final')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = json.loads((args.run / 'summary.json').read_text(encoding='utf-8'))
    conditions = json.loads((ROOT / 'docs/video_capture_conditions.json').read_text(encoding='utf-8'))
    mapping = {v: r['condition'] for r in conditions['reports'] for v in r['videos']}
    hashes = {Path(r['path']).relative_to(Path(summary['run']['input_directory'])).as_posix(): r['sha256'] for r in summary['run']['inputs']}
    for entry in [summary['run']['analyzer']] + summary['run']['tracking_modules']:
        assert digest(Path(entry['path'])) == entry['sha256'], entry['path']
    records, pages, duplicates = [], defaultdict(list), defaultdict(list)
    for result in summary['videos']:
        key = Path(result['video'])
        video = ROOT / 'soccer_videos' / key
        assert digest(video) == hashes[result['video']], video
        duplicates[hashes[result['video']]].append(result['video'])
        raw = result['raw_result']
        track = args.run / 'images' / key.parent / raw['tracking_data']
        data = json.loads(track.read_text(encoding='utf-8'))
        frames, vm = data['frames'], data['video_metadata']
        start = raw['tracking_summary']['shot_start_frame']
        if start is None:
            start = 0
        end = min(len(frames) - 1, start + round(vm['fps'] * .45))
        group = mapping.get(result['video'], 'real_primary' if key.parts[0] == 'real_test_videos' else 'other_test')
        selected, _ = select_flight(data['trajectory'], vm['fps'], vm['width'], vm['height'])
        trusted = {r['frame_number'] for r in selected}
        early_numbers = list(range(start, end + 1))
        observed_late = [r for r in frames if r['frame_number'] > end and r['observed']]
        csrt = [r for r in observed_late if r['source'] == 'csrt']
        late_numbers = sorted(set([r['frame_number'] for r in ([csrt[0], csrt[len(csrt) // 2], csrt[-1]] if csrt else [])]
                                  + [r['frame_number'] for r in ([observed_late[0], observed_late[len(observed_late) // 2], observed_late[-1]] if observed_late else [])]))
        cap = cv2.VideoCapture(str(video))
        early_path = args.output / group / key.with_suffix('') / 'early.jpg'
        late_path = args.output / group / key.with_suffix('') / 'late.jpg'
        panel(cap, frames, early_numbers, trusted, early_path, f'{result["video"]} | {LABELS[group]}',
              f'초기 f{start}~{end} 전체 {len(early_numbers)}프레임 | 3D 선택 {len(selected)} | 정답 속도 미검증')
        if late_numbers:
            panel(cap, frames, late_numbers, trusted, late_path, f'{result["video"]} | 초기 구간 이후',
                  '이후 관측 표본 / CSRT 첫·중간·끝 | 충돌·재반사 가능, 초기 속도 모델에 사용하지 않음')
        cap.release()
        sizes = [min(r['ball_bbox'][2] - r['ball_bbox'][0], r['ball_bbox'][3] - r['ball_bbox'][1]) for r in selected]
        counts = dict(Counter(frames[i]['source'] for i in early_numbers))
        record = {'video': result['video'], 'group': group, 'early_frames': early_numbers, 'late_sample_frames': late_numbers,
                  'trusted_frames': sorted(trusted), 'early_source_counts': counts,
                  'trusted_ball_minor_diameter_px': {'min': min(sizes), 'median': float(np.median(sizes)), 'max': max(sizes)} if sizes else None,
                  'existing_status': result['status'], 'existing_reason': result['failure_reason'],
                  'early_image': str(early_path), 'late_image': str(late_path) if late_numbers else None,
                  'visual_review_status': 'PENDING', 'speed_accuracy_verified': False}
        records.append(record)
        rel = early_path.relative_to(args.output).as_posix()
        late_rel = late_path.relative_to(args.output).as_posix()
        pages[group].append(f'<article><h3>{html.escape(result["video"])} · {result["status"]} {result["failure_reason"] or ""}</h3><a href="{rel}">초기 원본·박스 전체</a> '
                            + (f' / <a href="{late_rel}">이후 표본</a>' if late_numbers else '')
                            + f'<img loading="lazy" src="{rel}" alt="{html.escape(result["video"])}"></article>')
        print(f'{len(records)}/32 {result["video"]}: early={len(early_numbers)}, late={len(late_numbers)}, trusted={len(selected)}', flush=True)
    report = {'method': 'original_frames_and_existing_detection_boxes; visual_review_pending; no_speed_ground_truth',
              'group_counts': dict(Counter(r['group'] for r in records)), 'videos': records,
              'exact_duplicate_inputs': [keys for keys in duplicates.values() if len(keys) > 1]}
    (args.output / 'inspection.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    sections = ''.join(f'<details open><summary>{label}</summary>{"".join(pages[group])}</details>' for group, label in LABELS.items())
    (args.output / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>공 추적 그룹 검증</title><style>body{background:#111827;color:#eee;font:16px sans-serif;margin:24px}img{display:block;max-width:1280px;width:100%;margin:12px 0}a{color:#67e8f9}summary{font-size:24px;padding:18px}article{margin-bottom:40px}</style><h1>공 추적 그룹 검증</h1><p>원본 프레임과 검출 박스 대조. 공 위치·윤곽은 육안 검토 대상이며 속도 정확도를 보증하지 않습니다.</p>' + sections, encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'videos'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
