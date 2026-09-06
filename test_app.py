import os
import unittest

import plotly.graph_objects as go

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import app


def _collect_component_ids(component) -> set[str]:
    ids = set()
    component_id = getattr(component, "id", None)
    if component_id:
        # Pattern-matching ids (the comparison table's rows) are dicts.
        ids.add(component_id if isinstance(component_id, str) else repr(component_id))

    children = getattr(component, "children", None)
    if isinstance(children, (list, tuple)):
        for child in children:
            ids.update(_collect_component_ids(child))
    elif hasattr(children, "children"):
        ids.update(_collect_component_ids(children))

    return ids


class DashboardSmokeTests(unittest.TestCase):
    @staticmethod
    def _first_available_selection() -> tuple[str, str]:
        for target in app.TARGET_COLS:
            for option in app._available_model_types(target):
                if not option.get("disabled"):
                    return target, option["value"]
        raise AssertionError("No target/model combination is available for smoke testing.")

    def test_layout_exposes_core_panels(self) -> None:
        ids = _collect_component_ids(app.app.layout)
        self.assertTrue({"usa-map", "hover-panel", "streak-panel", "stats-panel"}.issubset(ids))

    def test_prediction_smoke(self) -> None:
        """Default view: stations only, no interpolated surface, scale still present."""
        target, model_type = self._first_available_selection()
        fig, status, title, subtitle, streak, stats, performance = app.run_prediction(
            1,
            False,
            True,
            target,
            model_type,
            "2026-05-01",
        )

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(status, "")
        # title/subtitle are styled components, so match against their JSON.
        self.assertIn(target, str(title.to_plotly_json()))
        self.assertIn(model_type, str(subtitle.to_plotly_json()))
        # Station trace + Iowa label + neighbour labels; no interpolated grid.
        self.assertGreaterEqual(len(fig.data), 2)
        self.assertNotIn("Interpolated grid", [trace.name for trace in fig.data])

        station_trace = next(t for t in fig.data if t.name == "Monitoring stations")
        self.assertTrue(station_trace.marker.showscale)

        for component in (streak, stats, performance):
            self.assertTrue(hasattr(component, "to_plotly_json"))

    def test_interpolation_toggle_adds_surface_and_keeps_scale(self) -> None:
        target, model_type = self._first_available_selection()
        fig, *_ = app.run_prediction(1, True, True, target, model_type, "2026-05-01")

        names = [trace.name for trace in fig.data]
        self.assertIn("Interpolated grid", names)

        # Exactly one colorbar, and it rides on the station trace either way.
        with_scale = [t.name for t in fig.data if getattr(t.marker, "showscale", False)]
        self.assertEqual(with_scale, ["Monitoring stations"])

    def test_landmarks_are_labelled_and_never_steal_a_hover(self) -> None:
        target, model_type = self._first_available_selection()
        fig, *_ = app.run_prediction(1, False, True, target, model_type, "2026-05-01")

        rendered = str(fig.to_plotly_json())
        for landmark in ("Des Moines", "Mississippi R.", "Clear Lake"):
            self.assertIn(landmark, rendered)

        # Landmark traces must be hover-inert, or they would shadow a station.
        for trace in fig.data:
            if trace.name != "Monitoring stations":
                self.assertEqual(trace.hoverinfo, "skip")
                self.assertFalse(trace.showlegend)

    def test_landmarks_can_be_turned_off(self) -> None:
        target, model_type = self._first_available_selection()
        fig, *_ = app.run_prediction(1, False, False, target, model_type, "2026-05-01")

        rendered = str(fig.to_plotly_json())
        self.assertNotIn("Des Moines", rendered)
        # The state labels are not landmarks and stay in both modes.
        self.assertIn("MINNESOTA", rendered)

    def test_missing_inputs_are_reported_not_crashed(self) -> None:
        outputs = app.run_prediction(0, True, True, None, None, None)

        self.assertEqual(len(outputs), 7)
        self.assertIn("Please pick", str(outputs[1].to_plotly_json()))

    def test_comparison_table_covers_every_scored_pair(self) -> None:
        table = app._comparison_table("pH", "Random Forest", "all")
        rendered = str(table.to_plotly_json())

        for model_type in app.MODEL_PREFIXES:
            self.assertIn(model_type, rendered)
        self.assertIn("Specific Conductance", rendered)

    def test_comparison_scope_filters_to_selected_target(self) -> None:
        table = app._comparison_table("pH", "Random Forest", "selected")
        rendered = str(table.to_plotly_json())

        self.assertIn("pH", rendered)
        self.assertNotIn("Specific Conductance", rendered)

    def test_build_feature_matrix_preserves_feature_contract(self) -> None:
        feature_matrix = app.build_feature_matrix(app.date(2026, 2, 14))

        self.assertEqual(feature_matrix.columns.tolist(), app.FEATURE_COLS)
        self.assertTrue((feature_matrix["doy"] == 45).all())
        self.assertEqual(len(feature_matrix), len(app.STATIONS))

    def test_target_assessment_missing_value(self) -> None:
        assessment = app._target_assessment("Nitrate", None)

        self.assertEqual(assessment["label"], "Unknown")
        self.assertEqual(assessment["color"], app.TEXT_MID)


if __name__ == "__main__":
    unittest.main()
