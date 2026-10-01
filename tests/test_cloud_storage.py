"""Tests for the GCS integration module. No real GCP project/credentials are
available in this project's development environment, so the
google.cloud.storage client is mocked — these verify the upload/download
logic makes the right calls with the right paths and degrades to a clean
no-op when GCS_BUCKET_NAME is unset, not that a real bucket round-trips
correctly. See storage.py's module docstring for that caveat.
"""
import importlib
from unittest.mock import MagicMock, call, patch

import student_journey.cloud.storage as gcs_storage


def test_disabled_by_default_when_no_bucket_configured(monkeypatch):
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    importlib.reload(gcs_storage)
    assert gcs_storage.is_enabled() is False


def test_upload_and_download_are_noops_when_disabled(tmp_path, monkeypatch):
    monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
    importlib.reload(gcs_storage)

    fake_model = tmp_path / "model_pipeline.joblib"
    fake_model.write_text("not a real model")

    # Should not raise even though no GCS client could ever be constructed.
    gcs_storage.upload_file(fake_model, "model_pipeline.joblib")
    downloaded = gcs_storage.download_file(tmp_path / "downloaded.joblib", "model_pipeline.joblib")
    assert downloaded is False

    uploaded_names = gcs_storage.upload_model_artifacts(tmp_path)
    assert uploaded_names == []


def test_enabled_when_bucket_configured(monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    importlib.reload(gcs_storage)
    try:
        assert gcs_storage.is_enabled() is True
        assert gcs_storage.GCS_BUCKET_NAME == "test-bucket"
    finally:
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        importlib.reload(gcs_storage)


def test_upload_file_calls_gcs_client_correctly(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    monkeypatch.setenv("GCS_MODEL_PREFIX", "models")
    importlib.reload(gcs_storage)
    try:
        local_file = tmp_path / "model_pipeline.joblib"
        local_file.write_text("fake model bytes")

        mock_blob = MagicMock()
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("student_journey.cloud.storage.storage.Client", return_value=mock_client):
            gcs_storage.upload_file(local_file, "model_pipeline.joblib")

        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.blob.assert_called_once_with("models/model_pipeline.joblib")
        mock_blob.upload_from_filename.assert_called_once_with(str(local_file))
    finally:
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        monkeypatch.delenv("GCS_MODEL_PREFIX", raising=False)
        importlib.reload(gcs_storage)


def test_download_file_returns_false_when_blob_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    importlib.reload(gcs_storage)
    try:
        mock_blob = MagicMock()
        mock_blob.exists.return_value = False
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("student_journey.cloud.storage.storage.Client", return_value=mock_client):
            result = gcs_storage.download_file(tmp_path / "out.joblib", "model_pipeline.joblib")

        assert result is False
        mock_blob.download_to_filename.assert_not_called()
    finally:
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        importlib.reload(gcs_storage)


def test_download_model_artifacts_skips_files_already_present_locally(tmp_path, monkeypatch):
    monkeypatch.setenv("GCS_BUCKET_NAME", "test-bucket")
    importlib.reload(gcs_storage)
    try:
        (tmp_path / "model_pipeline.joblib").write_text("already here")
        # The other MODEL_ARTIFACT_FILENAMES are NOT present locally.
        expected_missing = [f for f in gcs_storage.MODEL_ARTIFACT_FILENAMES if f != "model_pipeline.joblib"]

        mock_blob = MagicMock()
        mock_blob.exists.return_value = True
        mock_bucket = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket

        with patch("student_journey.cloud.storage.storage.Client", return_value=mock_client):
            downloaded = gcs_storage.download_model_artifacts(tmp_path)

        assert downloaded == expected_missing
        mock_bucket.blob.assert_has_calls([call(f"models/{f}") for f in expected_missing], any_order=True)
        assert mock_bucket.blob.call_count == len(expected_missing)
    finally:
        monkeypatch.delenv("GCS_BUCKET_NAME", raising=False)
        importlib.reload(gcs_storage)
        importlib.reload(gcs_storage)
