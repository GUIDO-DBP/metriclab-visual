import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from scripts.resource_io import install_package,sha256

class Distribution(unittest.TestCase):
    def package(self,root,files):
        archive=root/'models.zip'; inventory={'id':'models','files':{n:{'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()} for n,v in files.items()}}
        with zipfile.ZipFile(archive,'w') as z:
            z.writestr('PACKAGE_MANIFEST.json',json.dumps(inventory))
            for name,value in files.items(): z.writestr(name,value)
        return archive,{'id':'models','sha256':sha256(archive)}

    def test_additive_install_is_idempotent_and_validates_existing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a,e=self.package(root,{'runtime/models/a.bin':b'known model'})
            install_package(a,e,root); dest=root/'runtime/models/a.bin'; before=dest.stat().st_mtime_ns
            install_package(a,e,root); self.assertEqual(dest.stat().st_mtime_ns,before)
            dest.write_bytes(b'different')
            with self.assertRaises(ValueError): install_package(a,e,root)
            self.assertEqual(dest.read_bytes(),b'different')

    def test_traversal_and_non_runtime_paths_are_rejected(self):
        for path in ['../outside.bin','runtime/../../outside.bin','backend/main.py','runtime/C:bad','runtime\\bad.bin']:
            with self.subTest(path=path),tempfile.TemporaryDirectory() as temp:
                root=Path(temp); a,e=self.package(root,{path:b'bad'})
                with self.assertRaises(ValueError): install_package(a,e,root)
                self.assertFalse((root/'outside.bin').exists())

    def test_bad_archive_hash_or_conflict_makes_no_partial_install(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); a,e=self.package(root,{'runtime/new.bin':b'new','runtime/old.bin':b'old'})
            with self.assertRaises(ValueError): install_package(a,{**e,'sha256':'0'*64},root)
            (root/'runtime').mkdir(); (root/'runtime/old.bin').write_bytes(b'keep')
            with self.assertRaises(ValueError): install_package(a,e,root)
            self.assertFalse((root/'runtime/new.bin').exists())

if __name__=='__main__': unittest.main()
