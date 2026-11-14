#!/usr/bin/env bash
# Build a source snapshot locally; never reuse shared Windows build/devel.
# No dependency installation or upgrade, no ROS nodes and no vehicle outputs.
set -eo pipefail
task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
task_package_dir=$(cd -- "$task_script_dir/.." && pwd)
task_source_dir=$(cd -- "$task_package_dir/.." && pwd)
bash "$task_script_dir/check_original_environment.sh"
source /opt/ros/melodic/setup.bash
/usr/bin/python2 - "$task_package_dir/environment_baseline.json" <<'PY'
from __future__ import print_function
import json, sys, platform, subprocess
import numpy, cv2
with open(sys.argv[1]) as stream:
    baseline = json.load(stream)
assert platform.python_version() == baseline['python'], 'Python baseline mismatch'
assert platform.release() == baseline['kernel'], 'Kernel baseline mismatch'
assert numpy.__version__ == baseline['numpy'], 'NumPy baseline mismatch'
assert cv2.__version__ == baseline['opencv'], 'OpenCV baseline mismatch'
with open('/etc/os-release') as stream:
    release = stream.read()
assert ('PRETTY_NAME="' + baseline['ubuntu_pretty_name'] + '"') in release, 'Ubuntu baseline mismatch'
for package, expected in sorted(baseline['packages'].items()):
    actual = subprocess.check_output(['dpkg-query', '-W', '-f=${Version}', package]).strip()
    assert actual == expected, '%s baseline mismatch: %s != %s' % (package, actual, expected)
print('PASS: versions match recorded car baseline; no dependency changes needed')
PY
if [ "$(rosversion -d)" != melodic ]; then
  printf 'STOP: ROS is not the original Melodic.\n'
  exit 2
fi
if ! /usr/bin/python2 -c 'import sys; assert sys.version_info[:2] == (2, 7)'; then
  printf 'STOP: original Python2.7 not available.\n'
  exit 2
fi
task_workspace=$(mktemp -d "$HOME/nrt-validation-XXXXXX")
mkdir -p "$task_workspace/src"
printf '\nValidation workspace: %s\n' "$task_workspace"
dpkg-query -W -f='${Package}\t${Version}\n' | sort > "$task_workspace/packages-before.txt"
cp -a "$task_source_dir/mapping" "$task_workspace/src/mapping"
cp -a "$task_package_dir" "$task_workspace/src/acceleration_event"
(
  cd "$task_source_dir"
  find mapping acceleration_event -type f ! -name '*.pyc' ! -path '*/__pycache__/*' -print0 |
    sort -z | xargs -0 sha256sum
) > "$task_workspace/source-sha256.txt"
printf '\nBuilding only mapping and acceleration_event snapshots.\n'
(
  cd "$task_workspace"
  catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python2 -DCMAKE_BUILD_TYPE=Debug
  source "$task_workspace/devel/setup.bash"
  /usr/bin/python2 -B "$task_workspace/src/acceleration_event/tests/test_buffer_python2.py"
  # Import real node modules, without constructing any node/client/controller.
  PYTHONPATH="$task_workspace/src/acceleration_event/src:${PYTHONPATH:-}" /usr/bin/python2 -B - <<'PY'
from __future__ import print_function
import mapping.msg
import data_quality
import jiantu
import zhixian_realtime
import genzong_realtime
print('PASS: generated messages and all node modules import under original Python2')
PY
  roslaunch --nodes acceleration_event validation.launch
) 2>&1 | tee "$task_workspace/build-check.log"
dpkg-query -W -f='${Package}\t${Version}\n' | sort > "$task_workspace/packages-after.txt"
if ! cmp -s "$task_workspace/packages-before.txt" "$task_workspace/packages-after.txt"; then
  printf 'STOP: installed package inventory changed during validation. Inspect before continuing.\n'
  exit 3
fi
printf '\nPASS: build/import/launch parsing; installed package versions unchanged.\n'
printf 'Workspace retained: %s\n' "$task_workspace"
printf 'This is NOT live ROS replay or vehicle verification.\n'
