import subprocess


def run_and_output(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (stdout, stderr) = p.communicate()
    p_status = p.wait()

    if p_status != 0:
        print(stderr)
        print(f"Command {command} failed with exit code {p_status}")
        print(stdout)
        exit(p_status)

    return stdout.decode()
