"""
Unit and Integration Tests for Flask Web Application & Inference Pipeline
Verifies routes, single-reading inference, API JSON endpoint, and table rendering.
"""

from pathlib import Path
import sys
import unittest
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.app import app

class TestWebApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        app.config["TESTING"] = True

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Privacy-Preserving Cardiovascular Disease Classification", response.data)
        self.assertIn(b"FedAvg", response.data)

    def test_predict_page_get(self):
        response = self.client.get("/predict")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Heart Rate", response.data)
        self.assertIn(b"Systolic Blood Pressure", response.data)

    def test_predict_form_submission(self):
        # Test AMI high-risk profile
        data = {
            "model_choice": "federated_lr",
            "hr": 115.0,
            "sbp": 90.0,
            "dbp": 60.0,
            "spo2": 92.0
        }
        response = self.client.post("/predict", data=data)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Inference Diagnostic Assessment", response.data)
        self.assertIn(b"AMI Probability", response.data)

    def test_api_predict_endpoint(self):
        payload = {
            "hr": 80.0,
            "sbp": 120.0,
            "dbp": 80.0,
            "spo2": 98.0,
            "model": "federated_lr"
        }
        response = self.client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        result = json.loads(response.data.decode("utf-8"))
        self.assertEqual(result["status"], "success")
        self.assertIn("probability", result)
        self.assertIn("prediction", result)
        self.assertIn("disclaimer", result)
        self.assertTrue(0.0 <= result["probability"] <= 1.0)

    def test_comparison_page(self):
        response = self.client.get("/comparison")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Centralized vs Federated vs Isolated Hospitals", response.data)
        self.assertIn(b"roc_curves_comparison.png", response.data)

    def test_federated_page(self):
        response = self.client.get("/federated")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Decentralized Hospital Topology", response.data)
        self.assertIn(b"Communication Rounds Log", response.data)

if __name__ == "__main__":
    unittest.main()
