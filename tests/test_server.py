import collections
import csv
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import server


SENSOR_BODY = {
    'T_outside': 5.0,
    'T_inside':  20.0,
    'T_floor':   18.0,
    'SR_direct': 0.0,
}

LOG_BODY = {**SENSOR_BODY, 'fan_on': 0, 'heater_on': 1}


@pytest.fixture(autouse=True)
def _inject_mocks(tmp_path):
    """Replace global server state with mocks before every test."""
    mock_scaler        = MagicMock()
    mock_scaler.mean_  = np.array([5.0, 20.0, 18.0, 0.0])
    mock_scaler.scale_ = np.array([5.0, 3.0,  3.0,  100.0])
    mock_scaler.transform.return_value = np.zeros((1, 4))

    mock_mpc = MagicMock()
    mock_mpc.solve.return_value = 2   # heater only

    mock_ppo = MagicMock()
    mock_ppo.predict.return_value = (np.array(3), None)   # both on

    server._scaler     = mock_scaler
    server._mpc        = mock_mpc
    server._ppo        = mock_ppo
    server._window_buf = collections.deque(maxlen=24)
    server.LOG_PATH    = str(tmp_path / 'experience.csv')

    yield

    # Restore so module-level state doesn't leak
    server._scaler     = None
    server._mpc        = None
    server._ppo        = None
    server._window_buf = None


@pytest.fixture
def client():
    server.app.config['TESTING'] = True
    with server.app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /predict — structure
# ---------------------------------------------------------------------------

def test_predict_returns_200(client):
    r = client.post('/predict', json=SENSOR_BODY)
    assert r.status_code == 200


def test_predict_response_keys(client):
    r = client.post('/predict', json=SENSOR_BODY)
    data = r.get_json()
    assert set(data.keys()) == {'fan_on', 'heater_on', 'action', 'source'}


def test_predict_action_in_valid_range(client):
    r = client.post('/predict', json=SENSOR_BODY)
    assert r.get_json()['action'] in {0, 1, 2, 3}


def test_predict_fan_and_heater_are_binary(client):
    r = client.post('/predict', json=SENSOR_BODY)
    d = r.get_json()
    assert d['fan_on']    in {0, 1}
    assert d['heater_on'] in {0, 1}


# ---------------------------------------------------------------------------
# /predict — source routing
# ---------------------------------------------------------------------------

def test_predict_uses_mpc_when_random_low(client):
    with patch('numpy.random.random', return_value=0.1):   # 0.1 < 0.7 → MPC
        r = client.post('/predict', json=SENSOR_BODY)
    assert r.get_json()['source'] == 'mpc'


def test_predict_uses_ppo_when_random_high(client):
    with patch('numpy.random.random', return_value=0.9):   # 0.9 >= 0.7 → PPO
        r = client.post('/predict', json=SENSOR_BODY)
    assert r.get_json()['source'] == 'ppo'


def test_predict_mpc_action_matches_mock(client):
    with patch('numpy.random.random', return_value=0.1):
        r = client.post('/predict', json=SENSOR_BODY)
    assert r.get_json()['action'] == 2   # mock_mpc.solve returns 2


def test_predict_ppo_action_matches_mock(client):
    with patch('numpy.random.random', return_value=0.9):
        r = client.post('/predict', json=SENSOR_BODY)
    assert r.get_json()['action'] == 3   # mock_ppo.predict returns 3


# ---------------------------------------------------------------------------
# /predict — window buffer
# ---------------------------------------------------------------------------

def test_predict_appends_to_window_buffer(client):
    assert len(server._window_buf) == 0
    client.post('/predict', json=SENSOR_BODY)
    assert len(server._window_buf) == 1


def test_predict_window_fills_to_24(client):
    for _ in range(30):
        client.post('/predict', json=SENSOR_BODY)
    assert len(server._window_buf) == 24   # deque maxlen cap


# ---------------------------------------------------------------------------
# /log
# ---------------------------------------------------------------------------

def test_log_returns_200(client):
    r = client.post('/log', json=LOG_BODY)
    assert r.status_code == 200


def test_log_returns_ok_status(client):
    r = client.post('/log', json=LOG_BODY)
    assert r.get_json() == {'status': 'ok'}


def test_log_creates_csv_file(client):
    import os
    assert not os.path.exists(server.LOG_PATH)
    client.post('/log', json=LOG_BODY)
    assert os.path.exists(server.LOG_PATH)


def test_log_csv_has_correct_header(client):
    client.post('/log', json=LOG_BODY)
    with open(server.LOG_PATH) as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == server.LOG_FIELDS


def test_log_csv_appends_multiple_rows(client):
    client.post('/log', json=LOG_BODY)
    client.post('/log', json=LOG_BODY)
    with open(server.LOG_PATH) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


def test_log_csv_values_match_body(client):
    client.post('/log', json=LOG_BODY)
    with open(server.LOG_PATH) as f:
        row = next(csv.DictReader(f))
    assert float(row['T_outside']) == pytest.approx(5.0)
    assert float(row['T_inside'])  == pytest.approx(20.0)
    assert int(row['heater_on'])   == 1
