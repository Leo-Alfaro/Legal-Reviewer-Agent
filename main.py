import os
from crew_module import MyAgentsCrew

if __name__ == "__main__":
    print("Iniciando MyAgentsCrew...")
    input = {"path": os.path.abspath("./FILEPDF")}
    MyAgentsCrew().crew().kickoff(inputs=input)
    print("Fluxo finalizado. Verifique o arquivo report.md.")