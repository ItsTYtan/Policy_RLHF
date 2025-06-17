import subprocess

def get_top_level_packages():
    result = subprocess.run(["pip", "list", "--not-required", "--format=freeze"],
                            capture_output=True, text=True)
    packages = result.stdout.strip().splitlines()
    return packages

def write_requirements(packages, filename="requirements.txt"):
    with open(filename, "w") as f:
        f.write("\n".join(packages) + "\n")

if __name__ == "__main__":
    packages = get_top_level_packages()
    write_requirements(packages)
    print(f"✅ requirements.txt created with {len(packages)} top-level packages.")
