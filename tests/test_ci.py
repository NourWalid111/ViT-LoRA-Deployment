from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_required_project_files_exist():
    required_files = [
        "README.md",
        "requirements.txt",
        "requirements-docker.txt",
        "Dockerfile",
        "app/main.py",
        "app/model_loader.py",
        "app/preprocessing.py",
        "training/lora_model.py",
        "training/dataset.py",
    ]

    for file_path in required_files:
        assert (PROJECT_ROOT / file_path).exists(), (
            f"Required file is missing: {file_path}"
        )


def test_assignment_15_workflow_exists():
    workflow = PROJECT_ROOT / ".github" / "workflows" / "ci.yml"

    assert workflow.exists(), "GitHub Actions CI workflow is missing"


def test_project_structure_exists():
    assert (PROJECT_ROOT / "app").is_dir()
    assert (PROJECT_ROOT / "training").is_dir()
    assert (PROJECT_ROOT / "tests").is_dir()