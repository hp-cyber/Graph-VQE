import hashlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def test_shell_and_embedded_python_syntax(self):
        script=ROOT/'setup_local_cudaq.sh'
        subprocess.run(['bash','-n',str(script)],check=True)
        blocks=re.findall(r"<<'(PY[A-Z_]*)'\n(.*?)\n\1",script.read_text(),re.S)
        self.assertGreater(len(blocks),10)
        for tag,code in blocks:compile(code,str(script)+':'+tag,'exec')

    def test_extract_migrate_twice_and_verify_exact_sources(self):
        text=(ROOT/'setup_local_cudaq.sh').read_text()
        def block(name):
            section=text.split(name+'() {',1)[1]
            return section.split("<<'PYEOF'\n",1)[1].split('\nPYEOF',1)[0]
        with tempfile.TemporaryDirectory() as d:
            for repeat in range(2):
                subprocess.run([sys.executable,'-c',block('materialize_packaged_backend'),d],check=True,capture_output=True)
                subprocess.run([sys.executable,'-c',block('apply_packaged_backend_migrations'),str(Path(d)/'simulator_mps_dist.cu')],check=True,capture_output=True)
                for source in (ROOT/'backend/source').iterdir():
                    self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),hashlib.sha256((Path(d)/source.name).read_bytes()).hexdigest())
            source=(Path(d)/'simulator_mps_dist.cu').read_text()
            self.assertIn('hbatchWorkerGpuIds',source)
            self.assertIn('hbatchGeneration = 0',source)


if __name__=='__main__':unittest.main()
