"""One command that takes the evaluation files all the way to what the public sees.

The steps were separate scripts run by hand in the right order, which is how a chart ends
up describing an older run than the table beside it. They now run as one chain, and any
step that fails stops the rest.

    python training/publish.py              # rebuild everything locally
    python training/publish.py --push       # and publish it

Order matters and is fixed: the tables are written from the evaluation files first, then
the curves and charts are drawn from the same files, then the charts are copied into the
repositories that carry them, and only then is anything pushed.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
SITE = os.path.abspath(os.path.join(ROOT, "..", "moonfrost-site"))
CHARTS = os.path.join(ROOT, "docs", "charts")

# which chart belongs on which model card. The base carries all three because pretraining
# is its story; the tunes carry only the comparison they appear in.
CARD_CHARTS = {
    "dist/moonfrost-777m-base": ["benchmarks.png", "training_loss.png", "scale.png"],
    "dist/moonfrost-777m-instruct-v1": ["benchmarks.png"],
    "dist/moonfrost-777m-instruct-v2": ["benchmarks.png"],
}

HF_REPOS = {
    "dist/moonfrost-777m-base": "whoashish115/Moonfrost-777M",
    "dist/moonfrost-777m-instruct-v1": "whoashish115/Moonfrost-777M-Instruct-v1",
    "dist/moonfrost-777m-instruct-v2": "whoashish115/Moonfrost-777M-Instruct-v2",
}


def run(command, cwd=ROOT):
    print(">", " ".join(command))
    result = subprocess.run(command, cwd=cwd)
    if result.returncode:
        sys.exit(f"failed: {' '.join(command)}")


def copy_charts():
    for folder, wanted in CARD_CHARTS.items():
        target = os.path.join(ROOT, folder, "charts")
        os.makedirs(target, exist_ok=True)
        for name in os.listdir(target):
            if name not in wanted:
                os.remove(os.path.join(target, name))
        for name in wanted:
            shutil.copy2(os.path.join(CHARTS, name), os.path.join(target, name))
        print("charts:", folder, wanted)


def push_git():
    """Both repositories stay at a single commit, so every publish amends and force-pushes."""
    for repo in (ROOT, SITE):
        run(["git", "add", "-A"], cwd=repo)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo).returncode:
            run(["git", "commit", "--amend", "--no-edit", "--quiet"], cwd=repo)
            run(["git", "push", "--force-with-lease", "origin", "main"], cwd=repo)
        else:
            print("nothing to commit:", os.path.basename(repo))


def push_hugging_face():
    from huggingface_hub import HfApi
    api = HfApi()
    for folder, repo in HF_REPOS.items():
        patterns = ["README.md", "generation_config.json"]
        patterns += [f"charts/{name}" for name in CARD_CHARTS[folder]]
        api.upload_folder(folder_path=os.path.join(ROOT, folder), repo_id=repo,
                          repo_type="model", allow_patterns=patterns,
                          delete_patterns=["charts/*"], commit_message="Moonfrost 777M")
        api.super_squash_history(repo_id=repo, repo_type="model",
                                 commit_message="Moonfrost 777M")
        print("published:", repo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--push", action="store_true",
                        help="publish to GitHub and Hugging Face as well")
    arguments = parser.parse_args()

    run([sys.executable, "training/sync_benchmarks.py"])
    run([sys.executable, "training/export_curves.py"])
    run([sys.executable, "training/make_charts.py"])
    copy_charts()

    if arguments.push:
        push_git()
        push_hugging_face()
    print("done")


if __name__ == "__main__":
    main()
