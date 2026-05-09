import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dvl_correction import (
    DvlDeadReckoningTrack,
    DvlTrackConfig,
    ImuSample,
    NavigationFusionPipeline,
    RawDvlMeasurement,
    SynchronizerConfig,
)
from dvl_correction.cli import main


class DeadReckoningAndCliTest(unittest.TestCase):
    def test_dvl_track_integrates_velocity_in_nav_frame(self) -> None:
        track = DvlDeadReckoningTrack(DvlTrackConfig(initial_position_nav_m=(1.0, 0.0, 0.0)))

        track.update(
            timestamp_s=0.0,
            velocity_body_m_s=(1.0, 0.0, 0.0),
            attitude_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )
        state = track.update(
            timestamp_s=2.0,
            velocity_body_m_s=(1.0, 0.0, 0.0),
            attitude_quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        )

        np.testing.assert_allclose(state.position_nav_m, np.array([3.0, 0.0, 0.0]))

    def test_pipeline_can_generate_position_from_dvl_velocity_track(self) -> None:
        pipeline = NavigationFusionPipeline(
            dvl_track=DvlDeadReckoningTrack(),
            config=SynchronizerConfig(max_delay_s=0.0),
        )

        outputs = []
        for timestamp_s in (0.0, 1.0, 2.0):
            outputs.extend(
                pipeline.add_imu_sample(
                    ImuSample(
                        timestamp_s=timestamp_s,
                        angular_velocity_rad_s=(0.0, 0.0, 0.0),
                        linear_acceleration_m_s2=(0.0, 0.0, 9.80665),
                    )
                )
            )
            outputs.extend(
                pipeline.add_raw_dvl_measurement(
                    RawDvlMeasurement(
                        timestamp_s=timestamp_s,
                        velocity_dvl_m_s=(1.0, 0.0, 0.0),
                        valid_beams=(True, True, True, True),
                    )
                )
            )
        outputs.extend(pipeline.flush())

        self.assertGreaterEqual(pipeline.diagnostics.dvl_track_updates, 2)
        self.assertTrue(any(output.dvl_update_applied for output in outputs))
        self.assertGreater(outputs[-1].position_m[0], 0.5)

    def test_cli_writes_outputs_and_diagnostics_from_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            input_path = tmp_path / "input.jsonl"
            output_path = tmp_path / "nav.csv"
            diagnostics_path = tmp_path / "diagnostics.json"
            input_path.write_text(
                '{"type":"imu","timestamp_s":0.0,"angular_velocity_rad_s":[0,0,0],'
                '"linear_acceleration_m_s2":[0,0,9.80665]}\n'
                '{"type":"dvl","timestamp_s":0.0,"velocity_dvl_m_s":[1,0,0],'
                '"valid_beams":[true,true,true,true]}\n'
                '{"type":"imu","timestamp_s":1.0,"angular_velocity_rad_s":[0,0,0],'
                '"linear_acceleration_m_s2":[0,0,9.80665]}\n'
                '{"type":"dvl","timestamp_s":1.0,"velocity_dvl_m_s":[1,0,0],'
                '"valid_beams":[true,true,true,true]}\n',
                encoding="utf-8",
            )

            exit_code = main(
                [
                    "--jsonl",
                    str(input_path),
                    "--output-csv",
                    str(output_path),
                    "--diagnostics-json",
                    str(diagnostics_path),
                    "--enable-dvl-track",
                ]
            )

            with output_path.open("r", encoding="utf-8") as output_file:
                rows = list(csv.DictReader(output_file))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertGreaterEqual(len(rows), 2)
        self.assertGreaterEqual(diagnostics["dvl_updates_applied"], 1)


if __name__ == "__main__":
    unittest.main()
