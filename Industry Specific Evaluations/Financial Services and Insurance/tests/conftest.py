"""Shared pytest setup for the FSI evaluation module.

* Puts the module root on ``sys.path`` so tests import ``fsi_eval`` and
  ``checks`` the same way the notebooks do.
* Skips tests marked ``bedrock`` unless ``FSI_EVAL_LIVE=1`` is set, because
  they call live Amazon Bedrock and cost money.
* Provides the shared fixture file and a default authenticated session.
"""

import os
import sys
from pathlib import Path

import pytest

# Module root: the folder that holds fsi_eval/, checks/ and data/.
MODULE_ROOT = Path(__file__).resolve().parent.parent
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))

from fsi_eval.systems_of_record import Session, load_fixtures  # noqa: E402

# Path to the shared systems-of-record fixture file.
FIXTURE_PATH = MODULE_ROOT / "data" / "systems_of_record.json"


def pytest_collection_modifyitems(config, items):
    """Skip live Bedrock tests unless the caller opted in with FSI_EVAL_LIVE=1."""
    if os.environ.get("FSI_EVAL_LIVE") == "1":
        return
    skip_live = pytest.mark.skip(reason="live Bedrock test; set FSI_EVAL_LIVE=1 to run")
    for item in items:
        if "bedrock" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture
def fixtures():
    """A fresh copy of the shared systems-of-record fixtures."""
    return load_fixtures(FIXTURE_PATH)


@pytest.fixture
def session():
    """Authenticated chat session for CUST-001, set out of band."""
    return Session(customer_id="CUST-001", session_id="sess-test-001", channel="chat")
