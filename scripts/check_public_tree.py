"""Check publishable files; honors .gitignore without staging or creating .git."""
import argparse
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
BANNED = {'.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp','.dcm','.dicom',
          '.nii','.gz','.npy','.npz','.pth','.pt','.ckpt','.pkl','.pickle',
          '.safetensors','.onnx','.h5','.pdf','.csv','.tsv','.json','.jsonl',
          '.so','.dll','.dylib','.log','.key','.pem','.bin','.zip','.tar'}


def check(root):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(['git','init','--bare','--quiet',tmp],check=True)
        cmd=['git','--git-dir',tmp,'--work-tree',str(root)]
        paths=subprocess.check_output(cmd+['ls-files','--others','--exclude-standard','-z'],cwd=root).split(b'\0')
    errors=[]; count=0
    for raw in paths:
        if not raw:continue
        rel=raw.decode();p=root/rel;count+=1
        if p.is_symlink() or p.suffix.lower() in BANNED or p.name=='.DS_Store':errors.append(f'Prohibited file: {rel}');continue
        if p.stat().st_size>10*1024*1024:errors.append(f'Oversized source file: {rel}')
        try:text=p.read_text()
        except UnicodeDecodeError:errors.append(f'Binary content: {rel}');continue
        for marker in ['BEGIN PRIVATE KEY','BEGIN OPENSSH PRIVATE KEY','BEGIN RSA PRIVATE KEY']:
            if marker in text and p.name!='check_public_tree.py':errors.append(f'Credential marker: {rel}')
        # Personal workstation/server paths must not be included in release code.
        import re
        if re.search(r'/(?:home|Users)/(?!(?:username|path)/)[^/\s]+/',text):errors.append(f'Personal absolute path: {rel}')
    print(f'Checked {count} publishable files (ignored files excluded).')
    for error in errors:print(error)
    return bool(errors)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',nargs='?',type=Path,default=ROOT)
    raise SystemExit(check(parser.parse_args().root.resolve()))
