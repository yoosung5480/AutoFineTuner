from pathlib import Path
import shutil

# 내부 유틸: 파일/디렉토리 안전 삭제
def safe_remove(path: Path) -> None:
    try:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)  # py>=3.8
    except FileNotFoundError:
        pass
    except Exception as e:
        print(f"[WARN] remove failed: {path} -> {e}")