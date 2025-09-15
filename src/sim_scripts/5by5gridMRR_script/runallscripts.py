import subprocess

scripts = [
    # "/home/kimjosy/LocReg_Regularization-1/Simulations/gen_spanreg_heatmap.py",
    # "/home/kimjosy/LocReg_Regularization-1/Simulations/generateheatmap2.py",
    # "/home/kimjosy/LocReg_Regularization-1/Simulations/generatemap3.py",
    "/home/kimjosy/LocReg_Regularization-1/Simulations/genmap4.py",
    "/home/kimjosy/LocReg_Regularization-1/Simulations/genmap5.py",
    "/home/kimjosy/LocReg_Regularization-1/Simulations/genmap6.py"
    # "/home/kimjosy/LocReg_Regularization-1/Simulations/genmap7.py"
    ]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python3", script])
