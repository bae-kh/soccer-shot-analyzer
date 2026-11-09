"""Reproducible synthetic geometry/scoring validation; no YOLO or field accuracy claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tests'))
from test_goal_measurement import fixture, RULES
from backend.goal_measurement import assess_physical_goal, score_shot
from backend.goal_entry import assess_goal_entry
from backend.trajectory_render import goal_entry_image
from tools.review_tracking_groups import font


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cases = [
        ('slow_inside', fixture(), 'ESTIMATED_GOAL', '공 전체 통과 · 여러 프레임'),
        ('oblique_inside', fixture((-.8, .8), tilted=True), 'ESTIMATED_GOAL', '비스듬한 골대 면 통과'),
        ('front_overlap', fixture((-1, -.8, -.5)), 'UNDETERMINED', '화면상 겹침 · 골대 앞'),
        ('center_only', fixture((-.5, -.1, 0, .05)), 'UNDETERMINED', '중심만 통과 · 공 전체는 미통과'),
        ('left_outside', fixture((-.5, .5), x=-.5), 'ESTIMATED_OUTSIDE', '왼쪽 기둥 바깥 통과'),
        ('above_crossbar', fixture((-.5, .5), y=-.5), 'ESTIMATED_OUTSIDE', '크로스바 위 통과'),
        ('post_boundary', fixture((-.5, .5), x=.12), 'UNDETERMINED', '기둥 경계 · 오차 고려 보류'),
        ('reverse', fixture((.5, 0, -.5)), 'UNDETERMINED', '골대 뒤에서 앞으로 이동'),
    ]
    for name, change, title in [
        ('missing_middle', {'observed': False, 'source': 'missing'}, '중간 관측 누락'),
        ('csrt_middle', {'source': 'csrt'}, 'CSRT 크기를 깊이 근거로 사용 안 함')]:
        case = fixture()
        case[0][2].update(change)
        cases.append((name, case, 'UNDETERMINED', title))
    results, summary_rows = [], []
    canvas = Image.new('RGB', (1400, 95 + len(cases)*235), '#0f172a')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), '골대 면·공 전체 통과 검증 | 합성 관측 / 실제 영상 정확도와 별개', font=font(25), fill='white')
    draw.text((20, 52), '골대 3 × 2 m · 공 반지름 0.11 m · 독립적인 구 투영 수식으로 관측 생성', font=font(18), fill='#94a3b8')
    for index, (name, case, expected, title) in enumerate(cases):
        rows, meta, camera, config = case
        physical = assess_physical_goal(rows, meta, camera, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'}, config)
        score = score_shot(physical, {'measurement_status': 'ESTIMATED', 'speed': 60}, RULES)
        assert physical['status'] == expected, (name, physical)
        if expected == 'ESTIMATED_GOAL':
            assert score['value'] == 80.
        elif expected == 'ESTIMATED_OUTSIDE':
            assert score['value'] == 0
        else:
            assert score['value'] is None
        corners = np.asarray(config['corners_px'])
        area = {'bbox': [float(corners[:,0].min()), float(corners[:,1].min()),
                         float(corners[:,0].max()), float(corners[:,1].max())],
                'confidence': 1., 'coordinate_space': 'original_pixels', 'origin': 'top_left'}
        entry = assess_goal_entry(rows, area, meta, {'kick_detected': True}, {'status': 'FIXED_CANDIDATE'})
        entry['physical_goal_status'] = physical['status']
        saved = {'domain': 'synthetic', 'video': name+'.mp4', 'raw_result': {'physical_goal': physical,
                 'goal_entry': entry, 'trajectory_image': name+'.jpg', 'goal_entry_image': name+'_goal.jpg',
                 'score': score['value'], 'score_status': score['status'],
                 'score_evaluation': score, 'measurement_status': 'ESTIMATED', 'speed': 60.,
                 'comment': score.get('comment', '골라인 통과를 판단할 관측 근거가 부족합니다.')}}
        summary_rows.append(saved)
        results.append({'case': name, 'expected': expected, 'passed': True, 'observations': rows,
                        'config': config, 'physical_goal': physical, 'score_evaluation': score})
        image = np.full((1080, 1920, 3), 32, np.uint8)
        cv2.polylines(image, [np.rint(config['corners_px']).astype(np.int32)], True, (255, 255, 0), 5)
        for row in rows:
            if row['observed']:
                x1,y1,x2,y2 = map(round, row['ball_bbox'])
                cv2.rectangle(image, (x1,y1), (x2,y2), (0,220,0), 3)
        image_dir = args.output/'images'
        image_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(image_dir/(name+'.jpg')), image)
        cv2.imwrite(str(image_dir/(name+'_goal.jpg')), goal_entry_image(image, rows, entry, physical, score))
        crop = image[180:880, 350:1550]
        preview = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((360,210))
        top = 95+index*235
        canvas.paste(preview, (15,top))
        draw.text((400, top+10), title, font=font(24), fill='white')
        draw.text((400, top+55), physical['status'], font=font(20), fill='#34d399' if expected == 'ESTIMATED_GOAL' else '#fbbf24')
        draw.text((400, top+90), physical['reason'], font=font(17), fill='#cbd5e1')
        draw.text((400, top+125), f"참고 점수: {score['value'] if score['value'] is not None else '미평가'} · 검증용 예시 기준", font=font(19), fill='#94a3b8')
        draw.text((400, top+165), f"연속 관측 프레임: {[r['frame_number'] for r in rows if r['observed']]}", font=font(17), fill='#94a3b8')
    canvas.save(args.output/'synthetic_goal_cases.jpg', quality=93)
    report = {'domain': 'synthetic', 'actual_video_tracking_test': False,
              'cases_passed': len(results), 'cases_total': len(results), 'cases': results,
              'source_sha256': hashlib.sha256((ROOT/'backend/goal_measurement.py').read_bytes()).hexdigest(),
              'speed_note': '60 km/h is a supplied scoring fixture, not a computed speed in this validation.'}
    for filename, data in [('goal_validation.json', report), ('summary.json', {'domain': 'synthetic', 'videos': summary_rows})]:
        (args.output/filename).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    example = {'goal_plane': {**fixture()[3], 'confirmed_by_user': False},
               'scoring_rules': {**RULES, 'confirmed_by_user': False, 'accept_conditional_estimates': False}}
    (args.output/'synthetic_config_example.json').write_text(json.dumps(example, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'domain': 'synthetic', 'cases_passed': len(results), 'output': str(args.output)}, indent=2))


if __name__ == '__main__':
    main()
