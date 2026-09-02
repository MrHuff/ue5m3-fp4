# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Import target for TorchTitan's ``experimental.custom_import`` setting."""

from ue5m3_fp4.integrations.torchtitan.registration import register_torchtitan

register_torchtitan()
