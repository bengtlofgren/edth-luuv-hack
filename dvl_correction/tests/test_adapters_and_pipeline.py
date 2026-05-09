import tempfile
import unittest
from pathlib import Path

import numpy as np

from dvl_correction import (
    ImuSample,
    NavigationFusionPipeline,
    RawDvlMeasurement,
    SynchronizerConfig,
    build_navigation_pipeline_from_json,
    iter_sensor_jsonl,
    load_imu_csv,
    load_raw_dvl_csv,
)


class AdaptersAndPipelineTest(unittest.TestCase):
    def test_loads_imu_and_dvl_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            imu_path = tmp_path / "imu.csv"
            dvl_path = tmp_path / "dvl.csv"
            imu_path.write_text(
                "timestamp_s,gx,gy,gz,ax,ay,az\n"
                "0.0,0.0,0.0,0.0,0.0,0.0,0.0\n",
                encoding="utf-8",
            )
            dvl_path.write_text(
                "timestamp_s,vx,vy,vz,pos_x,pos_y,pos_z,beam1_valid,beam2_valid,beam3_valid,beam4_valid\n"
                "0.0,0.2,0.0,0.0,1.0,0.0,0.0,true,true,true,true\n",
                encoding="utf-8",
            )

            imu_samples = load_imu_csv(imu_path)
            dvl_samples = load_raw_dvl_csv(dvl_path)

        self.assertEqual(len(imu_samples), 1)
        self.assertEqual(len(dvl_samples), 1)
        np.testing.assert_allclose(imu_samples[0].angular_velocity_rad_s, np.zeros(3))
        np.testing.assert_allclose(dvl_samples[0].velocity_dvl_m_s, np.array([0.2, 0.0, 0.0]))

    def test_loads_jsonl_sensor_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "log.jsonl"
            path.write_text(
                '{"type":"imu","timestamp_s":0.0,"angular_velocity_rad_s":[0,0,0],'
                '"linear_acceleration_m_s2":[0,0,0]}\n'
                '{"type":"dvl","timestamp_s":0.0,"velocity_dvl_m_s":[0.1,0,0],'
                '"valid_beams":[true,true,true,true]}\n',
                encoding="utf-8",
            )

            records = list(iter_sensor_jsonl(path))

        self.assertEqual([record_type for record_type, _ in records], ["imu", "dvl"])

    def test_pipeline_fuses_buffered_imu_and_dvl_in_timestamp_order(self) -> None:
        pipeline = NavigationFusionPipeline(config=SynchronizerConfig(max_delay_s=0.2))

        outputs = []
        outputs.extend(
            pipeline.add_imu_sample(
                ImuSample(
                    timestamp_s=0.0,
                    angular_velocity_rad_s=(0.0, 0.0, 0.0),
                    linear_acceleration_m_s2=(0.0, 0.0, 0.0),
                )
            )
        )
        outputs.extend(
            pipeline.add_imu_sample(
                ImuSample(
                    timestamp_s=0.2,
                    angular_velocity_rad_s=(0.0, 0.0, 0.0),
                    linear_acceleration_m_s2=(0.0, 0.0, 0.0),
                )
            )
        )
        outputs.extend(
            pipeline.add_raw_dvl_measurement(
                RawDvlMeasurement(
                    timestamp_s=0.1,
                    velocity_dvl_m_s=(1.0, 0.0, 0.0),
                    position_nav_m=(0.1, 0.0, 0.0),
                    valid_beams=(True, True, True, True),
                )
            )
        )
        outputs.extend(
            pipeline.add_imu_sample(
                ImuSample(
                    timestamp_s=0.4,
                    angular_velocity_rad_s=(0.0, 0.0, 0.0),
                    linear_acceleration_m_s2=(0.0, 0.0, 0.0),
                )
            )
        )
        outputs.extend(pipeline.flush())

        self.assertTrue(any(output.dvl_update_applied for output in outputs))
        self.assertGreater(outputs[-1].velocity_m_s[0], 0.9)

    def test_builds_pipeline_from_json_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            path.write_text(
                "{"
                '"initial_state":{"position_m":[1,2,3]},'
                '"kalman":{"gravity_nav_m_s2":[0,0,0]},'
                '"dvl_quality":{"allowed_modes":["bottom"],"max_velocity_m_s":10.0},'
                '"synchronizer":{"max_delay_s":0.0},'
                '"dvl_track":{"enabled":true,"initial_position_nav_m":[1,2,3]}'
                "}",
                encoding="utf-8",
            )

            pipeline = build_navigation_pipeline_from_json(path)

        self.assertEqual(pipeline.config.max_delay_s, 0.0)
        self.assertIsNotNone(pipeline.dvl_track)
        np.testing.assert_allclose(pipeline.kalman.state.position_m, np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
