# Copyright (c) 2026 Graphcore Ltd. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install public OLMES controls when the wrapper explicitly enables them."""

from ue5m3_fp4.olmes_runtime import install_from_environment

install_from_environment()
