import numpy as np
from ultralytics import YOLO


# Função para calcular ângulo entre 3 pontos (vértice em b)
def CalcAngulo(a, b, c):
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


# Função para printar as saídas de cada exercício
def printReturn(nome, angulo, minimo, maximo):
    status = "OK ✅" if minimo <= angulo <= maximo else "⚠ Fora do ideal"
    return f"{nome}: {angulo:.2f}° ({status})"


# Escolha o exercício a monitorar
# Opções: "supino", "agachamento", "terra", "leg45", "leg90", "puxada_alta", "abdutora", "adutora", "hack",
# "remada_maquina", "remada_baixa", "remada_alta"
exercicio_atual = "leg45"

# Carrega o modelo de pose
model = YOLO("yolo11n-pose.pt")

# COCO keypoints IDs
exercicios_ids = {
    "ombro_esq": 5, "cotovelo_esq": 7, "punho_esq": 9,
    "ombro_dir": 6, "cotovelo_dir": 8, "punho_dir": 10,
    "quadril_esq": 11, "joelho_esq": 13, "tornozelo_esq": 15,
    "quadril_dir": 12, "joelho_dir": 14, "tornozelo_dir": 16
}

# Executa a IA na câmera e mostra sua saída
results = model.predict(source=0, stream=True, show=True, verbose=False)

for fps in results:
    keypoints = fps.keypoints.xy

    if len(keypoints) > 0:
        pessoa0 = keypoints[0]
        coords = {nome: np.array(pessoa0[idx]) for nome, idx in exercicios_ids.items()}

        # Ângulos básicos
        ang_cotovelo_esq = CalcAngulo(coords["ombro_esq"], coords["cotovelo_esq"], coords["punho_esq"])
        ang_cotovelo_dir = CalcAngulo(coords["ombro_dir"], coords["cotovelo_dir"], coords["punho_dir"])

        ang_joelho_esq = CalcAngulo(coords["quadril_esq"], coords["joelho_esq"], coords["tornozelo_esq"])
        ang_joelho_dir = CalcAngulo(coords["quadril_dir"], coords["joelho_dir"], coords["tornozelo_dir"])

        ang_quadril_esq = CalcAngulo(coords["ombro_esq"], coords["quadril_esq"], coords["joelho_esq"])
        ang_quadril_dir = CalcAngulo(coords["ombro_dir"], coords["quadril_dir"], coords["joelho_dir"])

        ang_ombro_esq = CalcAngulo(coords["cotovelo_esq"], coords["ombro_esq"], coords["quadril_esq"])
        ang_ombro_dir = CalcAngulo(coords["cotovelo_dir"], coords["ombro_dir"], coords["quadril_dir"])

        # ------------------------------------------------ #

        if exercicio_atual == "supino":
            print("\n=== SUPINO ===")
            print(printReturn("Cotovelo Esq", ang_cotovelo_esq, 45, 90))
            print(printReturn("Cotovelo Dir", ang_cotovelo_dir, 45, 90))

        elif exercicio_atual == "agachamento":
            print("\n=== AGACHAMENTO ===")
            print(printReturn("Joelho Esq", ang_joelho_esq, 90, 120))
            print(printReturn("Joelho Dir", ang_joelho_dir, 90, 120))

        elif exercicio_atual == "terra":
            print("\n=== LEVANTAMENTO TERRA ===")
            print(printReturn("Quadril Esq", ang_quadril_esq, 100, 120))
            print(printReturn("Quadril Dir", ang_quadril_dir, 100, 120))

        elif exercicio_atual == "leg45":
            print("\n=== LEG PRESS ===")
            print(printReturn("Joelho Esq", ang_joelho_esq, 90, 120))
            print(printReturn("Joelho Dir", ang_joelho_dir, 90, 120))
            print(printReturn("Quadril Esq", ang_quadril_esq, 60, 80))
            print(printReturn("Quadril Dir", ang_quadril_dir, 60, 80))

        elif exercicio_atual == "leg90":
            print("\n=== LEG PRESS HORIZONTAL (90°) ===")
            print(printReturn("Joelho Esq", ang_joelho_esq, 90, 120))
            print(printReturn("Joelho Dir", ang_joelho_dir, 90, 120))
            print(printReturn("Quadril Esq", ang_quadril_esq, 60, 100))
            print(printReturn("Quadril Dir", ang_quadril_dir, 60, 100))

        elif exercicio_atual == "flexora":
            print("\n=== FLEXORA ===")
            print(printReturn("Joelho Esq", ang_joelho_esq, 0, 120))
            print(printReturn("Joelho Dir", ang_joelho_dir, 0, 120))

        elif exercicio_atual == "extensora":
            print("\n=== EXTENSORA ===")
            print(printReturn("Joelho Esq", ang_joelho_esq, 30, 90))
            print(printReturn("Joelho Dir", ang_joelho_dir, 30, 90))

        elif exercicio_atual == "puxada_alta":
            print("\n=== PUXADA ALTA FRONTAL ===")
            print(printReturn("Ombro Esq", ang_ombro_esq, 60, 90))
            print(printReturn("Ombro Dir", ang_ombro_dir, 60, 90))
            print(printReturn("Cotovelo Esq", ang_cotovelo_esq, 80, 100))
            print(printReturn("Cotovelo Dir", ang_cotovelo_dir, 80, 100))

        elif exercicio_atual == "abdutora":
            print("\n=== ABDUTORA ===")
            print(printReturn("Quadril Esq (Abdução)", ang_quadril_esq, 0, 40))
            print(printReturn("Quadril Dir (Abdução)", ang_quadril_dir, 0, 40))

        elif exercicio_atual == "adutora":
            print("\n=== ADUTORA ===")
            print(printReturn("Quadril Esq (Adução)", ang_quadril_esq, 0, 30))
            print(printReturn("Quadril Dir (Adução)", ang_quadril_dir, 0, 30))

        elif exercicio_atual == "hack":
            print("\n=== HACK SQUAT ===")
            print(printReturn("Joelho Esq", ang_joelho_esq, 90, 120))
            print(printReturn("Joelho Dir", ang_joelho_dir, 90, 120))
            print(printReturn("Quadril Esq", ang_quadril_esq, 60, 100))
            print(printReturn("Quadril Dir", ang_quadril_dir, 60, 100))

        elif exercicio_atual == "remada_maquina":
            print("\n=== REMADA MÁQUINA ===")
            print(printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100))
            print(printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100))
            print(printReturn("Ombro Esq", ang_ombro_esq, 20, 40))
            print(printReturn("Ombro Dir", ang_ombro_dir, 20, 40))

        elif exercicio_atual == "remada_baixa":
            print("\n=== REMADA MÁQUINA BAIXA (CLOSE GRIP) ===")
            print(printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100))
            print(printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100))
            print(printReturn("Ombro Esq", ang_ombro_esq, 0, 30))
            print(printReturn("Ombro Dir", ang_ombro_dir, 0, 30))


        elif exercicio_atual == "remada_alta":
            print("\n=== REMADA MÁQUINA ALTA (WIDE GRIP) ===")
            print(printReturn("Cotovelo Esq", ang_cotovelo_esq, 60, 100))
            print(printReturn("Cotovelo Dir", ang_cotovelo_dir, 60, 100))
            print(printReturn("Ombro Esq", ang_ombro_esq, 45, 60))
            print(printReturn("Ombro Dir", ang_ombro_dir, 45, 60))



