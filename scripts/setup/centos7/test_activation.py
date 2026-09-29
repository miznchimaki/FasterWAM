"""Offline shell regression tests; no package installs or ELF modifications.

Run with Python 3.10+ and the actual Bash version to validate, for example:
  python test_activation.py --bash /path/to/bash-4.2
  BASH_UNDER_TEST=/bin/bash python test_activation.py

BASH_COMPAT does not reproduce historical Bash implementation bugs. These tests
run generated activations, launchers, and child shells with the chosen binary.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace
import venv

RUNTIME = Path(__file__).with_name('runtime.py')
BASH = shutil.which(os.environ.get('BASH_UNDER_TEST', 'bash'))
KEYS = ('PATH PS1 VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_PYTHON UV_PROJECT_ENVIRONMENT '
        'UV_PYTHON_DOWNLOADS UV_LINK_MODE UV_INDEX PYTHONNOUSERSITE '
        'FASTERWAM_GLIBC_ROOT LD_LIBRARY_PATH LD_PRELOAD PYTHONHOME PYTHONPATH '
        'CC CXX CUDA_HOME MAGICK_HOME MUJOCO_GL PYOPENGL_PLATFORM VK_ICD_FILENAMES')
SNAPSHOT = '''
snapshot() {
    local name
    for name in ''' + KEYS + '''; do
        declare -p "$name" 2>/dev/null || printf 'UNSET %s\n' "$name"
    done
}
'''


class ActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not BASH:
            raise RuntimeError('Bash executable not found; use --bash /path/to/bash')
        version = subprocess.run([BASH, '--version'], check=True, text=True,
                                 capture_output=True).stdout.splitlines()[0]
        print('Testing:', version, flush=True)
        spec = importlib.util.spec_from_file_location('runtime_activation_test', RUNTIME)
        cls.runtime = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.runtime
        spec.loader.exec_module(cls.runtime)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fasterwam-shell-test-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "project with spaces and 'quote"
        self.root.mkdir()
        self.uv = self.root / 'original-uv'
        self.uv.write_text('#!/bin/sh\nprintf "%s\\n" "$UV_PYTHON"\n')
        self.uv.chmod(0o755)
        self.gcc = self.root / 'gcc'
        (self.gcc / 'bin').mkdir(parents=True)
        for name in ('gcc', 'g++'):
            shutil.copy2('/bin/true', self.gcc / 'bin' / name)
        self.states = {}
        for profile in ('core', 'libero'):
            state_dir = self.root / '.runtime/centos7' / profile
            private = state_dir / 'python-base'
            virtual = self.root / ('.venv' if profile == 'core' else '.venvs/libero')
            for path in (private / 'bin', virtual / 'bin', state_dir / 'bin'):
                path.mkdir(parents=True)
            shutil.copy2(sys.executable, private / 'bin/python3.10')
            (virtual / 'bin/python').symlink_to(private / 'bin/python3.10')
            project = self.root if profile == 'core' else self.root / 'environments/libero'
            project.mkdir(exist_ok=True, parents=True)
            (project / 'uv.lock').write_text('version = 1\nregistry = "https://pypi.org/simple"\n')
            (project / 'pyproject.toml').write_text('[project]\nname = "fixture"\nversion = "0.1"\n')
            state = dict(root=str(self.root), profile=profile, venv=str(virtual), private=str(private),
                         python=str(private / 'bin/python3.10'), source_prefix=sys.base_prefix,
                         glibc=str(self.root / 'glibc'), loader='/lib64/ld-linux-x86-64.so.2',
                         rpath=['/usr/lib/x86_64-linux-gnu'], uv=str(self.uv),
                         gcc_root=str(self.gcc), cxx_dir=str(self.gcc / 'lib64'),
                         ffmpeg=str(self.root / 'ffmpeg'), patchelf='/missing/not-used')
            (state_dir / 'state.json').write_text(json.dumps(state))
            self.states[profile] = state
        with mock.patch.dict(os.environ, {'FASTERWAM_GCC_ROOT': str(self.gcc),
                                         'FASTERWAM_CUDA_HOME': str(self.root / 'cuda'),
                                         'MAGICK_HOME': str(self.root / 'magick'),
                                         'MUJOCO_GL': 'osmesa', 'PYOPENGL_PLATFORM': 'osmesa'}):
            for state in self.states.values():
                self.runtime.write_shell_files(state)

    def activate(self, profile='core'):
        return self.root / '.runtime/centos7' / profile / 'activate.sh'

    def bash(self, body):
        env = os.environ.copy()
        for key in KEYS.split():
            if key != 'PATH':
                env.pop(key, None)
        for key in ('BASH_COMPAT', 'BASH_ENV', 'ENV'):
            env.pop(key, None)
        env.update(VIRTUAL_ENV_DISABLE_PROMPT='1')
        script = 'set -euo pipefail\n' + SNAPSHOT + body
        result = subprocess.run([BASH, '--noprofile', '--norc', '-c', script],
                                env=env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr + '\nSCRIPT:\n' + script)
        return result

    def test_exact_restore_set_unset_export_attributes_and_newlines(self):
        self.bash(r'''
UV_INDEX=$'not exported \'quoted\'\ntrailing\n'; export -n UV_INDEX
[[ "$UV_INDEX" == *"'"* ]]
export PYTHONPATH=''
export LD_LIBRARY_PATH=$'/old libs\nline2'
LD_PRELOAD='old preload'; export -n LD_PRELOAD
CC=''; export -n CC
PS1=$'prompt \'quoted\'\n'; export -n PS1
unset UV_PYTHON CUDA_HOME CXX MAGICK_HOME
before=$(snapshot)
source ''' + shlex.quote(str(self.activate())) + '''
[[ "$UV_PYTHON" == "$VIRTUAL_ENV/bin/python" ]]
[[ -z ${LD_LIBRARY_PATH+x} && -z ${LD_PRELOAD+x} && -z ${PYTHONPATH+x} ]]
UV_INDEX='changed while active'
deactivate
after=$(snapshot)
[[ "$after" == "$before" ]] || { printf '%s\n%s\n' "$before" "$after"; exit 10; }
! declare -F deactivate >/dev/null
''')

    def test_source_twice_has_one_activation_and_restores_original(self):
        command = shlex.quote(str(self.activate()))
        self.bash(f'''
before=$(snapshot)
source {command}
first=$(snapshot)
source {command}
[[ "$(snapshot)" == "$first" ]]
deactivate
[[ "$(snapshot)" == "$before" ]]
''')

    def test_snapshot_declared_inside_activation_is_visible_globally(self):
        # Bash 4.2 mishandles combined `declare -gA name=()` inside functions.
        # Assert the snapshot survives function return, before any deactivation.
        self.bash(f'''
before_path=$(declare -p PATH)
source {shlex.quote(str(self.activate()))}
declare -p _FASTERWAM_SAVED_DECL >/dev/null
[[ ${{#_FASTERWAM_SAVED_DECL[@]}} -gt 0 ]]
[[ "${{_FASTERWAM_SAVED_DECL[PATH]}}" == "$before_path" ]]
deactivate
[[ "$(declare -p PATH)" == "$before_path" ]]
''')

    def test_missing_snapshot_refuses_deactivation_without_further_changes(self):
        activate = shlex.quote(str(self.activate()))
        self.bash(f'''
source {activate}
active=$(snapshot)
unset _FASTERWAM_SAVED_DECL
if deactivate; then echo 'deactivate unexpectedly succeeded'; exit 20; fi
[[ "$(snapshot)" == "$active" ]]
declare -F deactivate >/dev/null
if source {activate}; then echo 'reactivation unexpectedly succeeded'; exit 21; fi
[[ "$(snapshot)" == "$active" ]]
declare -F deactivate >/dev/null
''')

    def test_switch_profiles_restores_original_baseline(self):
        self.bash(f'''
before=$(snapshot)
source {shlex.quote(str(self.activate()))}
source {shlex.quote(str(self.activate('libero')))}
[[ "$VIRTUAL_ENV" == {shlex.quote(self.states['libero']['venv'])} ]]
[[ "$PATH" != *{shlex.quote(self.states['core']['venv'] + '/bin')}* ]]
deactivate
[[ "$(snapshot)" == "$before" ]]
''')

    def test_previous_ordinary_venv_and_its_deactivate_are_restored(self):
        other = self.root / 'other-environment'
        venv.EnvBuilder(with_pip=False).create(other)
        self.bash(f'''
original=$(snapshot)
source {shlex.quote(str(other / 'bin/activate'))}
prior=$(snapshot)
prior_function=$(declare -f deactivate)
for iteration in 1 2; do
    source {shlex.quote(str(self.activate()))}
    deactivate
    [[ "$(snapshot)" == "$prior" ]]
    [[ "$(declare -f deactivate)" == "$prior_function" ]]
done
deactivate
[[ "$(snapshot)" == "$original" ]]
''')

    def test_launchers_use_venv_uv_python_without_changing_parent(self):
        state = self.states['core']
        runtime_dir = Path(state['private']).parent
        result = subprocess.run([BASH, runtime_dir / 'bin/uv', '--version'], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), state['venv'] + '/bin/python')
        result = subprocess.run([BASH, runtime_dir / 'run', sys.executable, '-c',
                                 'import os;print(os.environ["UV_PYTHON"])'],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), state['venv'] + '/bin/python')

    def test_child_shell_inheritance_cannot_deactivate_parent(self):
        activate = shlex.quote(str(self.activate()))
        self.bash(f'''
before=$(snapshot)
source {activate}
active=$(snapshot)
(
  deactivate
  [[ "$(snapshot)" == "$before" ]]
)
[[ "$(snapshot)" == "$active" ]]
"$BASH" --noprofile --norc -c 'set -euo pipefail; inherited="$VIRTUAL_ENV"; ! declare -F deactivate >/dev/null; source "$1"; deactivate; [[ "$VIRTUAL_ENV" == "$inherited" ]]' bash {activate}
[[ "$(snapshot)" == "$active" ]]
deactivate
[[ "$(snapshot)" == "$before" ]]
''')

    def test_refresh_changes_only_three_shell_files(self):
        state = self.states['core']
        directory = Path(state['private']).parent
        expected = {directory / 'activate.sh', directory / 'bin/uv', directory / 'run'}
        for path in expected:
            path.write_text('# old generated shell file\n')
        package = Path(state['venv']) / 'lib/python3.10/site-packages/existing_package.py'
        package.parent.mkdir(parents=True)
        package.write_text('VERSION = "unchanged"\n')
        def fingerprint():
            return {path: (hashlib.sha256(path.read_bytes()).hexdigest(),
                           path.stat().st_ino, path.stat().st_mtime_ns)
                    for path in self.root.rglob('*') if path.is_file()}
        before = fingerprint()
        # Only the server-specific Python prefix query is simulated. The state
        # ownership guards and actual atomic shell-file writes execute normally.
        with (mock.patch.object(self.runtime, 'capture', return_value=json.dumps(
                [state['venv'], state['private']])) as identity,
              mock.patch.object(self.runtime, 'patch', side_effect=AssertionError('ELF patch attempted'))):
            self.runtime.refresh_shell(SimpleNamespace(state=str(directory / 'state.json')))
        after = fingerprint()
        self.assertEqual(before.keys(), after.keys())
        self.assertEqual({path for path in before if before[path] != after[path]}, expected)
        identity.assert_called_once_with(str(Path(state['venv']) / 'bin/python'), '-I', '-B', '-c',
                                         'import sys,json; print(json.dumps([sys.prefix,sys.base_prefix]))')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--bash', default=os.environ.get('BASH_UNDER_TEST', 'bash'))
    options, remaining = parser.parse_known_args()
    BASH = shutil.which(options.bash)
    if BASH is None:
        parser.error(f'Bash executable not found: {options.bash}')
    unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)
