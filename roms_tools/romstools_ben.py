import pytest
import os,sys
#sys.path.append("/Users/blsaenz/data/scratch/roms-tools")
#os.chdir("/Users/blsaenz/data/scratch/roms-tools")

if __name__ == "__main__":
    
    # This is equivalent to running:
    # pytest tests/test_surface_forcing.py -v --cov=roms_tools
    exit_code = pytest.main([
        "/Users/blsaenz/data/scratch/roms-tools/roms_tools/tests/test_datasets",
        "-v",
        "--cov=roms_tools"
    ])

    print(f"Tests finished with exit code: {exit_code}")
