import numpy as np
from ultralytics import YOLO

# Função para calcular ângulo entre 3 pontos (vértice em b)
def calcular_angulo(a, b, c):
    ba = a - b
    bc = c - b
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

# Função para validar faixas seguras
def validar_faixa(nome, angulo, minimo, maximo):
    status = "OK ✅" if minimo <= angulo <= maximo else "⚠ Fora do ideal"
    return f"{nome}: {angulo:.2f}° ({status})"

# Escolha o exercício a monitorar
# Opções: "supino", "agachamento", "terra", "squat_clean", "overhead_squat", "push_press"
exercicio_atual = "agachamento"

# Carrega o modelo de pose
model = YOLO("yolo11n-pose.pt")

# COCO keypoints IDs
kp_ids = {
    "ombro_esq": 5, "cotovelo_esq": 7, "punho_esq": 9,
    "ombro_dir": 6, "cotovelo_dir": 8, "punho_dir": 10,
    "quadril_esq": 11, "joelho_esq": 13, "tornozelo_esq": 15,
    "quadril_dir": 12, "joelho_dir": 14, "tornozelo_dir": 16
}

# Pega o primeiro frame da câmera
results = model.predict(source=0, stream=True, show=True, verbose=False)

for fps in results:
    keypoints = fps.keypoints.xy

    if len(keypoints) > 0:
        pessoa0 = keypoints[0]
        coords = {nome: np.array(pessoa0[idx]) for nome, idx in kp_ids.items()}

        # Ângulos básicos
        ang_cotovelo_esq = calcular_angulo(coords["ombro_esq"], coords["cotovelo_esq"], coords["punho_esq"])
        ang_cotovelo_dir = calcular_angulo(coords["ombro_dir"], coords["cotovelo_dir"], coords["punho_dir"])

        ang_joelho_esq = calcular_angulo(coords["quadril_esq"], coords["joelho_esq"], coords["tornozelo_esq"])
        ang_joelho_dir = calcular_angulo(coords["quadril_dir"], coords["joelho_dir"], coords["tornozelo_dir"])

        ang_quadril_esq = calcular_angulo(coords["ombro_esq"], coords["quadril_esq"], coords["joelho_esq"])
        ang_quadril_dir = calcular_angulo(coords["ombro_dir"], coords["quadril_dir"], coords["joelho_dir"])
        
        ang_ombro_esq = calcular_angulo(coords["cotovelo_esq"], coords["ombro_esq"], coords["quadril_esq"])
        ang_ombro_dir = calcular_angulo(coords["cotovelo_dir"], coords["ombro_dir"], coords["quadril_dir"])

        # ------------------------------------------------ #
        if exercicio_atual == "supino":
            print("\n=== SUPINO ===")
            print(validar_faixa("Cotovelo Esq", ang_cotovelo_esq, 45, 90))
            print(validar_faixa("Cotovelo Dir", ang_cotovelo_dir, 45, 90))

        elif exercicio_atual == "agachamento":
            print("\n=== AGACHAMENTO ===")
            print(validar_faixa("Joelho Esq", ang_joelho_esq, 90, 120))
            print(validar_faixa("Joelho Dir", ang_joelho_dir, 90, 120))

        elif exercicio_atual == "terra":
            print("\n=== LEVANTAMENTO TERRA ===")
            print(validar_faixa("Quadril Esq", ang_quadril_esq, 100, 120))
            print(validar_faixa("Quadril Dir", ang_quadril_dir, 100, 120))

        elif exercicio_atual == "squat_clean":
            print("\n=== SQUAT CLEAN ===")
            print(validar_faixa("Joelho Esq", ang_joelho_esq, 80, 120))
            print(validar_faixa("Joelho Dir", ang_joelho_dir, 80, 120))
            print(validar_faixa("Cotovelo Esq", ang_cotovelo_esq, 60, 120))
            print(validar_faixa("Cotovelo Dir", ang_cotovelo_dir, 60, 120))

        elif exercicio_atual == "overhead_squat":
            print("\n=== OVERHEAD SQUAT ===")
            print(validar_faixa("Ombro Esq", ang_ombro_esq, 150, 180))
            print(validar_faixa("Ombro Dir", ang_ombro_dir, 150, 180))
            print(validar_faixa("Joelho Esq", ang_joelho_esq, 90, 120))
            print(validar_faixa("Joelho Dir", ang_joelho_dir, 90, 120))

        elif exercicio_atual == "push_press":
            print("\n=== PUSH PRESS ===")
            print(validar_faixa("Joelho Esq (Dip)", ang_joelho_esq, 100, 120))
            print(validar_faixa("Joelho Dir (Dip)", ang_joelho_dir, 100, 120))
            print(validar_faixa("Cotovelo Esq (Press)", ang_cotovelo_esq, 0, 45))
            print(validar_faixa("Cotovelo Dir (Press)", ang_cotovelo_dir, 0, 45))