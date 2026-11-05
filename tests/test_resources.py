import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import unittest
import venv
import zipfile

REPO = Path(__file__).resolve().parents[1]


def run_python(code, root, cwd):
    env = dict(os.environ, PYTHONPATH=str(root), HF_CACHE_REMOTE=str(cwd / 'cache'))
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise AssertionError(result.stdout + result.stderr)
    return result.stdout


class ResourceTests(unittest.TestCase):
    def test_builtin_paths_work_outside_checkout_and_with_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            code = '''
from pathlib import Path
from unittest.mock import patch
from configs.slurm import SLURM_CONFIGS
from scripts.cli import EvalConfig, _render_and_submit
for profile in [None, *SLURM_CONFIGS]:
    output = Path.cwd() / str(profile)
    config = EvalConfig(model_name_or_path='example/model', model_output_dir=str(output),
                        tasks=['amalia_arc_challenge_chat_mt_pt'], slurm_config=profile)
    with patch('builtins.input', return_value='n'), patch('subprocess.run', side_effect=AssertionError('submission')):
        _render_and_submit(config)
    script, = output.glob('*.j2_*')
    rendered = script.read_text()
    assert 'amalia_arc_challenge_chat_mt_pt' in rendered
from scripts.cli import HarnessArgs
path = Path(HarnessArgs().include_path)
assert path.is_absolute() and (path / 'arc_mt/arc_challenge_chat_mt_pt.yaml').is_file(), path
'''
            run_python(code, REPO, cwd)

    def test_explicit_relative_paths_survive_profile_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / 'custom.j2').write_text('{{ harness_args.include_path }} {{ tasks[0] }}')
            code = '''
from pathlib import Path
from unittest.mock import patch
from configs.slurm import SLURM_CONFIGS
from scripts.cli import EvalConfig, HarnessArgs, _render_and_submit
for profile in SLURM_CONFIGS:
    config = EvalConfig(model_name_or_path='example/model', model_output_dir=profile,
                        tasks=['example_task'], slurm_config=profile, templates_file='custom.j2',
                        harness_args=HarnessArgs(include_path='custom-tasks'))
    with patch('builtins.input', return_value='n'), patch('subprocess.run', side_effect=AssertionError('submission')):
        _render_and_submit(config)
    assert (Path(profile) / 'custom.j2_example_task').read_text() == 'custom-tasks example_task'
'''
            run_python(code, REPO, cwd)


class DistributionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        source = cls.root / 'source'
        source.mkdir()
        for name in ['configs', 'tasks', 'scripts', 'inspect', 'tasks_external', 'tests',
                     'pyproject.toml', 'README.md', 'LICENSE.md']:
            path = REPO / name
            if path.is_dir():
                shutil.copytree(path, source / name, ignore=shutil.ignore_patterns('__pycache__'))
            else:
                shutil.copy2(path, source / name)
        if (REPO / "MANIFEST.in").is_file():
            shutil.copy2(REPO / "MANIFEST.in", source / "MANIFEST.in")
        for relative in ['vibe_check/extra.yaml', 'alba/processed_alba/extra.json',
                         'alba/alba_original/extra.json']:
            local = source / 'tasks/amalia-bench' / relative
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text('task: local_only\n')
            (local.parent / 'private_helper.py').write_text('local_only = True\n')
        task = source / 'tasks/amalia-bench/humaneval_pt'
        for filename in ['.env', '.env.local']:
            (task / filename).write_text('AUDIT_MARKER=not-a-real-secret\n')
        offline = source / 'tasks/amalia-bench/ruler_pt/dataset/offline_fixture.json'
        offline.write_text('{}\n')
        direct_dir = cls.root / 'direct-wheel'
        direct_dir.mkdir()
        run_python(
            'from setuptools.build_meta import build_wheel; '
            f'build_wheel({str(direct_dir)!r})', source, source,
        )
        with zipfile.ZipFile(next(direct_dir.glob('*.whl'))) as bundle:
            cls.direct_members = set(bundle.namelist())
        stale = source / 'build/lib/tasks/amalia-bench/obsolete.yaml'
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text('task: obsolete_task\n')
        archive_dir = cls.root / 'archives'
        archive_dir.mkdir()
        run_python(
            'from setuptools.build_meta import build_sdist; '
            f'build_sdist({str(archive_dir)!r})', source, source,
        )
        archive, = archive_dir.glob('*.tar.gz')
        with tarfile.open(archive) as bundle:
            cls.sdist_members = {member.name.split('/', 1)[1] for member in bundle.getmembers()
                                 if member.isfile() and '/' in member.name}
            bundle.extractall(archive_dir, filter='data')
        unpacked, = (p for p in archive_dir.iterdir() if p.is_dir())
        cls.sdist_source = unpacked
        run_python(
            'from setuptools.build_meta import build_wheel; '
            f'build_wheel({str(archive_dir)!r})', unpacked, unpacked,
        )
        wheel, = archive_dir.glob('*.whl')
        cls.wheel = wheel
        cls.installed = cls.root / 'installed'
        with zipfile.ZipFile(wheel) as bundle:
            cls.members = set(bundle.namelist())
            bundle.extractall(cls.installed)

    def test_distributions_exclude_dotenv_but_keep_offline_data(self):
        private = {'tasks/amalia-bench/humaneval_pt/' + name for name in ['.env', '.env.local']}
        for label, members in [('direct wheel', self.direct_members),
                               ('sdist wheel', self.members), ('sdist', self.sdist_members)]:
            with self.subTest(artifact=label):
                self.assertEqual(private & members, set())
                self.assertIn('tasks/amalia-bench/ruler_pt/dataset/offline_fixture.json', members)

    def test_installed_console_loads_caller_dotenv(self):
        prefix = self.root / 'console-env'
        venv.EnvBuilder(with_pip=True, symlinks=True).create(prefix)
        packages = Path(sysconfig.get_path('purelib', vars={'base': str(prefix), 'platbase': str(prefix)}))
        (packages / 'test_dependencies.pth').write_text(sysconfig.get_path('purelib') + '\n')
        env = {key: value for key, value in os.environ.items()
               if key not in {'HF_CACHE_REMOTE', 'ACTIVATE_ENV', 'PYTHONPATH', 'PYTHON_DOTENV_DISABLED'}}
        env.update(PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK='1')
        install = subprocess.run(
            [str(prefix / 'bin/python'), '-m', 'pip', 'install', '--no-index', '--no-deps', str(self.wheel)],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(install.returncode, 0, install.stdout + install.stderr)
        caller = self.root / 'caller'
        nested = caller / 'nested'
        nested.mkdir(parents=True)
        (caller / '.env').write_text('HF_CACHE_REMOTE=/scratch/file-cache\nACTIVATE_ENV=file-env\n')
        command = [str(prefix / 'bin/amalia'), '--model-name-or-path', 'example/model',
                   '--tasks', 'amalia_arc_challenge_chat_mt_pt', '--slurm-config', 'marenostrum']
        for label, cwd, overrides, cache, activation in [
            ('file', caller, {}, '/scratch/file-cache', 'file-env'),
            ('ancestor', nested, {}, '/scratch/file-cache', 'file-env'),
            ('exported', caller, {'HF_CACHE_REMOTE': '/scratch/exported-cache', 'ACTIVATE_ENV': 'exported-env'},
             '/scratch/exported-cache', 'exported-env'),
        ]:
            with self.subTest(case=label):
                output = self.root / ('console-' + label)
                result = subprocess.run(
                    command + ['--model-output-dir', str(output)], cwd=cwd, env=env | overrides,
                    input='n\n', capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                script, = output.glob('*.j2_*')
                rendered = script.read_text()
                self.assertIn('export HF_HOME="' + cache + '/huggingface"', rendered)
                self.assertIn('ACT_ENV="' + activation + '"', rendered)
                self.assertIn('Skipped submission', result.stdout)
        empty = self.root / 'no-configuration'
        empty.mkdir()
        (empty / '.env').write_text('')
        result = subprocess.run(
            command + ['--model-output-dir', str(self.root / 'console-missing')], cwd=empty, env=env,
            input='n\n', capture_output=True, text=True, timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('needs HF_CACHE_REMOTE', result.stderr)

    def test_runtime_assets_survive_sdist_and_wheel(self):
        excluded = {'tasks/amalia-bench/LegalBenchPT/original/trash.txt',
                    'tasks/amalia-bench/mmlu_mt/translation_cache.json'}
        local_dirs = ('tasks/amalia-bench/vibe_check/',
                      'tasks/amalia-bench/alba/processed_alba/',
                      'tasks/amalia-bench/alba/alba_original/')
        assets = {p.relative_to(REPO).as_posix() for p in (REPO / 'tasks/amalia-bench').rglob('*')
                  if p.is_file() and p.suffix != '.pyc' and '__pycache__' not in p.parts}
        assets = {path for path in assets if path not in excluded and not path.startswith(local_dirs)}
        self.assertEqual(sorted(assets - self.members), [])
        self.assertNotIn("tasks/amalia-bench/obsolete.yaml", self.members)
        for relative in ["vibe_check/extra.yaml", "alba/processed_alba/extra.json",
                         "alba/alba_original/extra.json"]:
            self.assertNotIn("tasks/amalia-bench/" + relative, self.members)

    def test_direct_wheel_excludes_local_development_files(self):
        self.assertIn('tasks/amalia-bench/saudade-pt/utils.py', self.direct_members)
        direct = {name for name in self.direct_members if '.dist-info/' not in name}
        via_sdist = {name for name in self.members if '.dist-info/' not in name}
        self.assertEqual(direct, via_sdist)
        for relative in ['vibe_check', 'alba/processed_alba', 'alba/alba_original']:
            prefix = 'tasks/amalia-bench/' + relative + '/'
            self.assertEqual([name for name in self.direct_members if name.startswith(prefix)], [])

    def test_installed_tasks_match_source_distribution(self):
        code = '''
import json
from pathlib import Path
from lm_eval.tasks import TaskManager
import tasks.configs
root = Path(tasks.configs.__file__).resolve().parent / 'amalia-bench'
manager = TaskManager(include_path=str(root), include_defaults=False)
print(json.dumps(sorted(manager.all_tasks)))
'''
        source = json.loads(run_python(code, self.sdist_source, self.root).splitlines()[-1])
        installed = json.loads(run_python(code, self.installed, self.root).splitlines()[-1])
        self.assertIn('amalia_arc_challenge_chat_mt_pt', installed)
        self.assertEqual(installed, source)

    def test_installed_path_with_apostrophe_renders_valid_shell(self):
        installed = self.root / "o'brien"
        shutil.copytree(self.installed, installed)
        code = """
from pathlib import Path
import shlex
import subprocess
from unittest.mock import patch
from scripts.cli import EvalConfig, HarnessArgs, _render_and_submit
config = EvalConfig(model_name_or_path='example/model', model_output_dir='quoted-output',
                    tasks=['amalia_arc_challenge_chat_mt_pt'])
with patch('builtins.input', return_value='n'), patch('subprocess.run', side_effect=AssertionError('submission')):
    _render_and_submit(config)
script, = Path('quoted-output').glob('*.j2_*')
subprocess.run(['bash', '-n', str(script)], check=True)
line, = [line for line in script.read_text().splitlines() if '--include_path' in line]
arguments = shlex.split(line.split('+=(', 1)[1].removesuffix(')'))
assert arguments == ['--include_path', HarnessArgs().include_path]
"""
        run_python(code, installed, self.root)

    def test_installed_ruler_vocabulary_and_book_work_outside_checkout(self):
        code = """
import importlib.util
from pathlib import Path
from configs.resources import BENCHMARK_PATH
from unittest.mock import patch
for name, word in [('cwe_utils', 'casa'), ('prepare_niah', 'grande-casa')]:
    path = Path(BENCHMARK_PATH) / 'ruler_pt' / f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    with patch('nltk.download', return_value=True):
        spec.loader.exec_module(module)
    assert word in module.WORDS
    if name == 'prepare_niah':
        assert 'PREFÁCIO' in module.get_haystack('essay')
"""
        run_python(code, self.installed, self.root)

    def test_installed_ruler_qa_reads_local_data_outside_checkout(self):
        code = """
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
from configs.resources import BENCHMARK_PATH
hotpot = [{'context': [['titulo', ['texto']]], 'question': 'pergunta', 'answer': 'resposta'}]
for variant in ['ruler_pt', 'ruler_en']:
    path = Path(BENCHMARK_PATH) / variant / 'qa_utils.py'
    spec = importlib.util.spec_from_file_location(variant, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    filename = ('pt_hotpotqa_distractor_validation_transformed.json' if variant == 'ruler_pt'
                else 'hotpot_dev_distractor_v1.json')
    dataset = path.parent / 'dataset'
    dataset.mkdir(exist_ok=True)
    (dataset / filename).write_text(json.dumps(hotpot))
    with patch.object(module, 'download_json', side_effect=OSError('offline')):
        questions, documents = module.read_hotpotqa()
    assert documents == ['titulo\\ntexto']
    assert questions == [{'query': 'pergunta', 'outputs': ['resposta'], 'context': [0]}]
"""
        run_python(code, self.installed, self.root)

    def test_installed_config_includes_and_launcher(self):
        code = '''
from pathlib import Path
from unittest.mock import patch
from lm_eval.utils import load_yaml_config
from scripts.cli import EvalConfig, HarnessArgs, _render_and_submit
root = Path(HarnessArgs().include_path)
for filename in ['calame/calame_handwritten.yaml', 'alba_mcq/default/alba_syntax.yaml',
                 'mmlu_mt/default/mmlu_prehistory.yaml', 'saudade-pt/saudade_mcq_pt.yaml']:
    config = load_yaml_config(str(root / filename))
    assert config['dataset_path']
config = EvalConfig(model_name_or_path='example/model', model_output_dir='rendered',
                    tasks=['amalia_arc_challenge_chat_mt_pt'])
with patch('builtins.input', return_value='n'), patch('subprocess.run', side_effect=AssertionError('submission')):
    _render_and_submit(config)
script, = Path('rendered').glob('*.j2_*')
assert str(root) in script.read_text()
'''
        run_python(code, self.installed, self.root)


if __name__ == '__main__':
    unittest.main()
