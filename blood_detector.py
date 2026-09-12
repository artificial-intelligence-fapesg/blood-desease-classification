"""
blood_detector.py
==================

Módulo responsável por carregar e executar os modelos de classificação
de doenças a partir de exames de sangue (hemograma + glicemia + pressão
arterial), classificando o paciente em uma de 4 classes:

    "Saudável", "Hipertenso", "Leucemia", "Diabetes"

Suporta:
1. Uso de um modelo específico (dentre vários cadastrados).
2. Votação (ensemble) entre múltiplos modelos, combinando modelos com
   pontos fortes distintos (ex.: um com melhor acurácia, outro com
   melhor precisão, outro com melhor F1-score) — mesmo padrão usado em
   `pneumonia_detector.py`.

Variáveis de entrada (nessa ordem — `FEATURE_ORDER`):
    idade               (anos)
    hemoglobina         (g/dL)
    hematocrito         (%)
    hemacias            (milhões/mm³)
    leucocitos          (células/mm³)
    plaquetas           (em milhares/mm³, ex.: 150 = 150.000/mm³)
    glicemia            (mg/dL)
    pressaoSistolica    (mmHg)
    pressaoDiastolica   (mmHg)

COMO INTEGRAR SEUS MODELOS REAIS (.h5 / .keras, TensorFlow 2.16+)
------------------------------------------------------------------
A classe `BloodDiseaseDetector` sabe carregar e executar redes neurais
Keras salvas em `.h5`, `.hdf5` ou `.keras`, com saída softmax de 4
neurônios (uma probabilidade por classe, na mesma ordem de `CLASSES`).

O jeito mais simples de integrar (sem editar nenhum código):

1. Salve o modelo treinado com o **mesmo nome** de uma das chaves de
   `MODEL_CATALOG` (ex.: `ModeloA_AltaAcuracia.h5`).
2. Copie o arquivo para a pasta `models_sangue/`.
3. (Opcional) Se o seu modelo espera as features padronizadas
   (`StandardScaler`/`MinMaxScaler` do scikit-learn), salve o objeto
   `scaler` treinado com `joblib.dump(scaler, "models_sangue/<nome>.scaler.pkl")`
   — ele será carregado e aplicado automaticamente antes da inferência.
4. Reinicie o app. `build_model_registry()` detecta os arquivos
   automaticamente e passa a usar o modelo real no lugar do simulado
   (`MockBloodDiseaseDetector`) — a interface não precisa de nenhuma
   alteração.

IMPORTANTE: o modelo real deve ter sido treinado com as classes na
mesma ordem de `CLASSES` (["Saudável", "Hipertenso", "Leucemia",
"Diabetes"]) e as features na mesma ordem de `FEATURE_ORDER`. Se a
ordem das classes do seu `LabelEncoder`/`OneHotEncoder` for diferente,
ajuste a lista `CLASSES` abaixo (ou reordene no treinamento) para que
o índice de saída do softmax corresponda à posição correta.

Enquanto nenhum arquivo `.h5`/`.keras` correspondente for encontrado em
`models_sangue/`, o registro usa `MockBloodDiseaseDetector`, que simula
o modelo com regras clínicas simplificadas + ruído (útil para testar a
interface antes de o treinamento estar pronto).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


CLASSES = ["Diabetes", "Hipertenso", "Leucemia", "Saudável"]

# Ordem das features exigida pelos modelos (deve casar com o treinamento).
FEATURE_ORDER = [
    "idade",
    "hemoglobina",
    "hematocrito",
    "hemacias",
    "leucocitos",
    "plaquetas",
    "glicemia",
    "pressaoSistolica",
    "pressaoDiastolica",
]

# Faixas de referência aproximadas — usadas apenas para exibir ajuda na
# interface (help text) e para o mock simular um comportamento clínico
# plausível. NÃO usar como critério diagnóstico definitivo.
REFERENCE_RANGES = {
    "idade": "anos",
    "hemoglobina": "g/dL (referência adulto: 12–17)",
    "hematocrito": "% (referência adulto: 36–50)",
    "hemacias": "milhões/mm³ (referência adulto: 4.0–6.0)",
    "leucocitos": "células/mm³ (referência: 4.000–11.000)",
    "plaquetas": "em milhares/mm³ (referência: 150–450)",
    "glicemia": "mg/dL em jejum (referência: 70–99)",
    "pressaoSistolica": "mmHg (referência: <120)",
    "pressaoDiastolica": "mmHg (referência: <80)",
}


# ---------------------------------------------------------------------------
# Estruturas de resultado
# ---------------------------------------------------------------------------
@dataclass
class DiagnosisResult:
    """Resultado padronizado de um único modelo."""

    model_name: str
    label: str                     # uma das classes em CLASSES
    confidence: float              # confiança da classe prevista (0-1)
    probabilities: dict            # {"Saudável": 0.05, "Hipertenso": 0.02, ...}


@dataclass
class EnsembleResult:
    """Resultado consolidado da votação entre múltiplos modelos."""

    label: str
    confidence: float
    probabilities: dict                            # probabilidades médias (soft-vote)
    individual_results: list[DiagnosisResult] = field(default_factory=list)
    votes: dict = field(default_factory=dict)       # {"Saudável": 1, "Leucemia": 2, ...}
    unanimous: bool = False


PROCESSING_STEPS = [
    "Validando faixas e integridade dos exames laboratoriais...",
    "Normalizando variáveis hematológicas, glicêmicas e pressóricas...",
    "Executando a rede neural de classificação...",
    "Calculando probabilidades por classe...",
    "Consolidando laudo assistido por IA...",
]


# ---------------------------------------------------------------------------
# Compatibilidade entre versões do Keras (reaproveitado do detector de RX)
# ---------------------------------------------------------------------------
def _strip_unknown_layer_kwargs(config, layers_module) -> None:
    """Percorre recursivamente um dict de configuração de modelo Keras e
    remove, de cada camada, os parâmetros que o construtor da camada
    instalada localmente não reconhece (ex.: `quantization_config`)."""
    import inspect

    if isinstance(config, dict):
        if "class_name" in config and isinstance(config.get("config"), dict):
            layer_cls = getattr(layers_module, config["class_name"], None)
            if layer_cls is not None:
                try:
                    accepted = set(inspect.signature(layer_cls.__init__).parameters.keys())
                except (TypeError, ValueError):
                    accepted = None
                if accepted:
                    accepted |= {"name", "trainable", "dtype"}
                    layer_config = config["config"]
                    for key in list(layer_config.keys()):
                        if key not in accepted:
                            layer_config.pop(key, None)
        for value in config.values():
            _strip_unknown_layer_kwargs(value, layers_module)
    elif isinstance(config, list):
        for item in config:
            _strip_unknown_layer_kwargs(item, layers_module)


def _load_h5_with_sanitized_config(model_path: Path):
    """Carrega um .h5 removendo, da configuração das camadas, parâmetros
    que a versão local do Keras não reconhece, preservando os pesos."""
    import json
    import os
    import shutil
    import tempfile
    import h5py
    import tensorflow as tf

    with h5py.File(str(model_path), "r") as f:
        raw_config = f.attrs.get("model_config")
        if raw_config is None:
            raise RuntimeError(
                "O arquivo .h5 não contém a chave 'model_config' — ele não "
                "parece ter sido salvo com `model.save(...)` do Keras."
            )
        if isinstance(raw_config, bytes):
            raw_config = raw_config.decode("utf-8")

    config_dict = json.loads(raw_config)
    _strip_unknown_layer_kwargs(config_dict, tf.keras.layers)
    sanitized_json = json.dumps(config_dict)

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".h5")
    os.close(tmp_fd)
    try:
        shutil.copyfile(str(model_path), tmp_path)
        with h5py.File(tmp_path, "r+") as f:
            del f.attrs["model_config"]
            f.attrs["model_config"] = sanitized_json

        return tf.keras.models.load_model(tmp_path, compile=False)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Classe base — todo modelo (real ou simulado) implementa esta interface
# ---------------------------------------------------------------------------
class BaseBloodModel:
    name: str = "modelo-base"
    metrics: dict = {}

    def predict(self, patient_df: pd.DataFrame) -> DiagnosisResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Implementação para modelos reais (.h5 / .keras — TensorFlow 2.16+ / Keras 3)
# ---------------------------------------------------------------------------
class BloodDiseaseDetector(BaseBloodModel):
    """
    Wrapper de um modelo real de classificação de doenças sanguíneas,
    treinado em TensorFlow/Keras e salvo em `.h5`, `.hdf5` ou `.keras`,
    com saída softmax de 4 neurônios (uma por classe de `CLASSES`).
    """

    SUPPORTED_EXTENSIONS = {".h5", ".hdf5", ".keras"}

    def __init__(
        self,
        name: str,
        model_path: str,
        metrics: dict,
        scaler_path: Optional[str] = None,
        feature_order: Optional[list[str]] = None,
    ):
        """
        Args:
            name: nome de exibição do modelo (deve casar com uma chave
                de MODEL_CATALOG).
            model_path: caminho para o arquivo .h5/.hdf5/.keras.
            metrics: dicionário de métricas (acurácia, precisão, recall, f1).
            scaler_path: caminho opcional para um `StandardScaler`/
                `MinMaxScaler` do scikit-learn salvo com `joblib.dump`.
                Se fornecido (ou se existir um arquivo
                `<model_path sem extensão>.scaler.pkl`), as features
                são padronizadas com esse scaler antes da inferência.
            feature_order: ordem das colunas esperada pelo modelo. Por
                padrão usa `FEATURE_ORDER`.
        """
        self.name = name
        self.model_path = Path(model_path)
        self.metrics = metrics
        self.feature_order = feature_order or FEATURE_ORDER
        self.model = None
        self.scaler = None
        self._scaler_path = Path(scaler_path) if scaler_path else self.model_path.with_suffix("").with_suffix(".scaler.pkl")
        self._load_model()
        self._load_scaler()

    # 1) Carregamento do modelo -------------------------------------------------
    def _load_model(self):
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Arquivo do modelo '{self.name}' não encontrado em: {self.model_path}\n"
                "Verifique se o caminho está correto e se o arquivo foi copiado "
                "para a pasta 'models_sangue/'."
            )

        if self.model_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Extensão '{self.model_path.suffix}' não suportada para o modelo "
                f"'{self.name}'. Formatos aceitos: {', '.join(sorted(self.SUPPORTED_EXTENSIONS))}."
            )

        try:
            import tensorflow as tf
        except ImportError as e:
            raise ImportError(
                "TensorFlow não está instalado. Instale com "
                "`pip install tensorflow>=2.16` para carregar modelos .h5/.keras "
                "(veja requirements.txt)."
            ) from e

        try:
            self.model = tf.keras.models.load_model(str(self.model_path), compile=False)
        except TypeError as e:
            if "Unrecognized keyword arguments" in str(e):
                print(
                    f"[modelos] '{self.name}': o .h5 foi salvo com uma versão do "
                    "Keras diferente da instalada (parâmetro de camada não "
                    "reconhecido). Tentando carregar em modo de compatibilidade..."
                )
                try:
                    self.model = _load_h5_with_sanitized_config(self.model_path)
                    print(f"[modelos] '{self.name}': carregado com sucesso em modo de compatibilidade.")
                except Exception as fallback_error:
                    raise RuntimeError(
                        f"Falha ao carregar o modelo '{self.name}' a partir de "
                        f"'{self.model_path}', mesmo em modo de compatibilidade.\n"
                        f"Erro original: {e}\n"
                        f"Erro no modo de compatibilidade: {fallback_error}\n\n"
                        "Sugestão: reexporte o modelo com a mesma versão de "
                        "TensorFlow/Keras instalada neste ambiente e salve novamente."
                    ) from fallback_error
            else:
                raise RuntimeError(
                    f"Falha ao carregar o modelo '{self.name}' a partir de "
                    f"'{self.model_path}'.\nErro original: {e}"
                ) from e
        except Exception as e:
            raise RuntimeError(
                f"Falha ao carregar o modelo '{self.name}' a partir de "
                f"'{self.model_path}'.\nErro original: {e}\n\n"
                "Dicas de compatibilidade (TensorFlow 2.16+ / Keras 3):\n"
                "  - Se o .h5 foi salvo com uma versão muito antiga do "
                "TensorFlow/Keras, reexporte o modelo com TF 2.16+ e salve "
                "novamente (.h5 ou .keras).\n"
                "  - Se o modelo usa camadas, losses ou métricas customizadas, "
                "passe-as via `custom_objects` em "
                "`tf.keras.models.load_model(path, compile=False, "
                "custom_objects={...})` (ajuste `_load_model`)."
            ) from e

    # 2) Carregamento do scaler (opcional) --------------------------------------
    def _load_scaler(self):
        if self._scaler_path and self._scaler_path.exists():
            try:
                import joblib
                self.scaler = joblib.load(self._scaler_path)
                print(f"[modelos] '{self.name}': scaler carregado de '{self._scaler_path.name}'.")
            except Exception as e:
                print(
                    f"[modelos] Aviso: falha ao carregar o scaler de '{self._scaler_path}' "
                    f"para o modelo '{self.name}': {e}. Prosseguindo sem padronização."
                )
                self.scaler = None

    # 3) Pré-processamento -------------------------------------------------------
    def _preprocess(self, patient_df: pd.DataFrame) -> np.ndarray:
        ordered = patient_df[self.feature_order].astype(float)
        array = ordered.to_numpy()
        if self.scaler is not None:
            array = self.scaler.transform(array)
        return array

    # 4) Inferência ---------------------------------------------------------------
    def predict(self, patient_df: pd.DataFrame) -> DiagnosisResult:
        if self.model is None:
            raise RuntimeError(f"Modelo '{self.name}' não carregado.")

        processed = self._preprocess(patient_df)
        raw_output = self.model.predict(processed, verbose=0)
        raw_output = np.asarray(raw_output).reshape(-1)

        if raw_output.size != len(CLASSES):
            raise ValueError(
                f"O modelo '{self.name}' retornou {raw_output.size} saída(s), "
                f"mas eram esperadas {len(CLASSES)} (uma por classe: {CLASSES})."
            )

        # Garante que a saída seja uma distribuição de probabilidade válida
        # (caso o modelo não tenha softmax na última camada).
        if not np.isclose(raw_output.sum(), 1.0, atol=1e-3):
            exp = np.exp(raw_output - raw_output.max())
            raw_output = exp / exp.sum()

        probabilities = {cls: float(p) for cls, p in zip(CLASSES, raw_output)}
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]

        return DiagnosisResult(
            model_name=self.name,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )


# ---------------------------------------------------------------------------
# Implementação simulada (para testes de UI enquanto os modelos reais não
# estão prontos). Usa regras clínicas simplificadas para gerar "logits"
# plausíveis a partir dos exames e aplica viés/ruído próprios de cada
# modelo simulado, para que a votação produza divergências realistas.
# ---------------------------------------------------------------------------
def _rule_based_logits(row: dict) -> dict:
    """
    Gera logits (não normalizados) plausíveis para cada classe a partir
    de regras clínicas simplificadas. NÃO representa um critério
    diagnóstico real — serve apenas para simular o comportamento de um
    modelo enquanto a rede neural real não está integrada.
    """
    idade = float(row.get("idade", 0) or 0)
    hemoglobina = float(row.get("hemoglobina", 0) or 0)
    leucocitos = float(row.get("leucocitos", 0) or 0)
    plaquetas = float(row.get("plaquetas", 0) or 0)
    glicemia = float(row.get("glicemia", 0) or 0)
    pas = float(row.get("pressaoSistolica", 0) or 0)
    pad = float(row.get("pressaoDiastolica", 0) or 0)

    logits = {"Saudável": 1.2, "Hipertenso": 0.0, "Leucemia": 0.0, "Diabetes": 0.0}

    # --- Diabetes (glicemia de jejum) ---
    if glicemia >= 126:
        logits["Diabetes"] += 3.2
        logits["Saudável"] -= 1.2
    elif glicemia >= 100:
        logits["Diabetes"] += 1.2

    # --- Hipertensão (pressão arterial) ---
    if pas >= 140 or pad >= 90:
        logits["Hipertenso"] += 3.2
        logits["Saudável"] -= 1.2
    elif pas >= 130 or pad >= 85:
        logits["Hipertenso"] += 1.2

    # --- Leucemia (leucocitose acentuada, plaquetopenia, anemia) ---
    if leucocitos >= 30000:
        logits["Leucemia"] += 4.2
        logits["Saudável"] -= 2.0
    elif leucocitos >= 11000:
        logits["Leucemia"] += 1.6
        logits["Saudável"] -= 0.3

    if plaquetas < 150:
        logits["Leucemia"] += 0.9
    if plaquetas < 100:
        logits["Leucemia"] += 1.2

    if hemoglobina < 12:
        logits["Leucemia"] += 0.5

    # Leve efeito de idade (fatores de risco aumentam com a idade)
    if idade >= 60:
        logits["Hipertenso"] += 0.3
        logits["Diabetes"] += 0.2
        logits["Leucemia"] += 0.2

    return logits


def _softmax(logits: dict) -> dict:
    classes = list(logits.keys())
    values = np.array([logits[c] for c in classes], dtype=np.float64)
    exp = np.exp(values - values.max())
    probs = exp / exp.sum()
    return {c: float(p) for c, p in zip(classes, probs)}


class MockBloodDiseaseDetector(BaseBloodModel):
    def __init__(self, name: str, metrics: dict, bias: dict = None, noise: float = 0.35):
        self.name = name
        self.metrics = metrics
        # viés (em logit) que este modelo simulado aplica a cada classe,
        # para diferenciar o "comportamento" de cada modelo na votação.
        self.bias = bias or {c: 0.0 for c in CLASSES}
        self.noise = noise

    def predict(self, patient_df: pd.DataFrame) -> DiagnosisResult:
        time.sleep(0.25)  # simula custo computacional da inferência

        row = patient_df.iloc[0].to_dict()
        base_logits = _rule_based_logits(row)

        # Ruído determinístico a partir dos valores do paciente + nome do
        # modelo, para que o mesmo paciente sempre produza o mesmo
        # resultado-base, mas cada modelo simulado varie de forma
        # consistente (útil para comparar execuções).
        seed_source = "|".join(f"{k}:{row.get(k)}" for k in FEATURE_ORDER) + f"|{self.name}"
        digest = hashlib.sha256(seed_source.encode("utf-8")).hexdigest()
        rng = random.Random(int(digest[:8], 16))

        final_logits = {
            c: base_logits[c] + self.bias.get(c, 0.0) + rng.uniform(-self.noise, self.noise)
            for c in CLASSES
        }
        probabilities = {c: round(p, 4) for c, p in _softmax(final_logits).items()}
        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]

        return DiagnosisResult(
            model_name=self.name,
            label=label,
            confidence=confidence,
            probabilities=probabilities,
        )


# ---------------------------------------------------------------------------
# Registro de modelos (Model Registry)
# ---------------------------------------------------------------------------
class ModelRegistry:
    """
    Mantém a lista de modelos disponíveis na interface e implementa a
    lógica de seleção de modelo único ou votação (ensemble).
    """

    def __init__(self):
        self._models: dict[str, BaseBloodModel] = {}

    def register(self, model: BaseBloodModel):
        self._models[model.name] = model

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def get_metrics(self, name: str) -> dict:
        return self._models[name].metrics

    def get_all_metrics(self) -> dict[str, dict]:
        return {name: m.metrics for name, m in self._models.items()}

    def predict_single(self, model_name: str, patient_df: pd.DataFrame) -> DiagnosisResult:
        if model_name not in self._models:
            raise ValueError(f"Modelo '{model_name}' não está registrado.")
        return self._models[model_name].predict(patient_df)

    def predict_ensemble(
        self,
        patient_df: pd.DataFrame,
        model_names: Optional[list[str]] = None,
        method: str = "soft",
    ) -> EnsembleResult:
        """
        Executa a votação entre múltiplos modelos.

        method="soft"  -> decide pela média das probabilidades de cada
                           classe (mais sensível à confiança de cada
                           modelo).
        method="hard"  -> decide pela classe mais votada (maioria
                           simples); em caso de empate, usa o soft-vote
                           como desempate.
        Em ambos os casos, o detalhe do voto de cada modelo é retornado
        para transparência do laudo.
        """
        names = model_names or self.list_models()
        individual_results = [self._models[name].predict(patient_df) for name in names]

        votes = {c: 0 for c in CLASSES}
        for r in individual_results:
            votes[r.label] += 1
        classes_with_votes = [c for c, v in votes.items() if v > 0]
        unanimous = len(classes_with_votes) <= 1

        avg_probabilities = {
            c: round(sum(r.probabilities[c] for r in individual_results) / len(individual_results), 4)
            for c in CLASSES
        }

        if method == "hard":
            max_votes = max(votes.values())
            top_classes = [c for c, v in votes.items() if v == max_votes]
            if len(top_classes) == 1:
                label = top_classes[0]
            else:
                # empate -> desempata pela probabilidade média (soft-vote)
                label = max(top_classes, key=lambda c: avg_probabilities[c])
        else:  # soft (padrão)
            label = max(avg_probabilities, key=avg_probabilities.get)

        confidence = avg_probabilities[label]

        return EnsembleResult(
            label=label,
            confidence=confidence,
            probabilities=avg_probabilities,
            individual_results=individual_results,
            votes=votes,
            unanimous=unanimous,
        )


# ---------------------------------------------------------------------------
# Catálogo de modelos disponíveis
# ---------------------------------------------------------------------------
# ATUALIZE este catálogo com as métricas reais de cada modelo, calculadas
# em um conjunto de teste independente (métricas macro-médias, já que o
# problema é multiclasse). Os valores devem estar entre 0 e 1.
MODEL_CATALOG = {
    "VitaBL98A": {
        "descricao": "Otimizado para melhor acurácia geral entre as 4 classes. Bom equilíbrio entre as categorias.",
        "acuracia": 0.98,
        "precisao": 0.99,
        "recall": 0.97,
        "f1_score": 0.96,
        "especificidade": 0.96,
    },
    "VitaBL99A": {
        "descricao": "Otimizado para minimizar falsos positivos (maior precisão macro). Mais conservador ao classificar doenças.",
        "acuracia": 0.99,
        "precisao": 0.99,
        "recall": 0.99,
        "f1_score": 0.98,
        "especificidade": 0.97,
    },
    "VitaBLI": {
        "descricao": "Otimizado para o melhor equilíbrio entre precisão e recall (F1 macro). Bom para triagem geral.",
        "acuracia": 0.99,
        "precisao": 0.99,
        "recall": 0.99,
        "f1_score": 0.99,
        "especificidade": 0.99,
    },
}

ENSEMBLE_LABEL = "Votação (Ensemble de Modelos)"

MODELS_DIR = Path(__file__).parent / "models_sangue"


def _find_model_file(name: str) -> Optional[Path]:
    """
    Procura, na pasta `models_sangue/`, um arquivo de modelo real cujo
    nome (sem extensão) case com `name`. Aceita .h5, .hdf5 e .keras,
    nessa ordem de prioridade.
    """
    if not MODELS_DIR.exists():
        return None
    for ext in (".h5", ".hdf5", ".keras"):
        candidate = MODELS_DIR / f"{name}{ext}"
        if candidate.exists():
            return candidate
    return None


def build_model_registry() -> ModelRegistry:
    """
    Monta o registro de modelos usado pela interface.

    Para cada modelo do `MODEL_CATALOG`, procura automaticamente um
    arquivo `models_sangue/<nome_do_modelo>.h5` (ou `.hdf5`/`.keras`),
    além de um scaler opcional `models_sangue/<nome_do_modelo>.scaler.pkl`.
    Se encontrar o modelo, carrega o modelo real (`BloodDiseaseDetector`);
    caso contrário, usa um modelo simulado (`MockBloodDiseaseDetector`)
    baseado em regras clínicas, para que a interface continue
    funcionável antes do treinamento estar pronto.

    Ou seja: basta treinar o modelo, salvá-lo como
    `models_sangue/ModeloA_AltaAcuracia.h5` (mesmo nome da chave do
    catálogo) e reiniciar o app — nenhuma edição de código é necessária.
    """

    registry = ModelRegistry()

    # Perfis usados apenas pelos modelos simulados (mock), para diferenciar
    # o comportamento de cada "modelo" enquanto o real não está disponível.
    # `bias` desloca o logit de cada classe (valores positivos tornam o
    # modelo mais propenso a apontar aquela classe).
    mock_profiles = {
        "ModeloA_AltaAcuracia": {
            "bias": {"Saudável": 0.0, "Hipertenso": 0.0, "Leucemia": 0.0, "Diabetes": 0.0},
            "noise": 0.30,
        },
        "ModeloB_AltaPrecisao": {
            "bias": {"Saudável": 0.3, "Hipertenso": -0.15, "Leucemia": -0.15, "Diabetes": -0.15},
            "noise": 0.30,
        },
        "ModeloC_AltoF1": {
            "bias": {"Saudável": -0.15, "Hipertenso": 0.1, "Leucemia": 0.15, "Diabetes": 0.1},
            "noise": 0.35,
        },
    }

    for name, metrics in MODEL_CATALOG.items():
        model_file = _find_model_file(name)

        if model_file is not None:
            try:
                scaler_candidate = MODELS_DIR / f"{name}.pkl"
                registry.register(
                    BloodDiseaseDetector(
                        name=name,
                        model_path=model_file,
                        metrics=metrics,
                        scaler_path=scaler_candidate if scaler_candidate.exists() else None,
                    )
                )
                print(f"[modelos] '{name}': modelo real carregado de '{model_file.name}'.")
                continue
            except Exception as e:
                print(
                    f"[modelos] Aviso: falha ao carregar o modelo real '{name}' "
                    f"a partir de '{model_file}': {e}\n"
                    f"          Usando modelo simulado (mock) no lugar."
                )
                profile = mock_profiles.get(name, {"bias": {c: 0.0 for c in CLASSES}, "noise": 0.35})
                registry.register(MockBloodDiseaseDetector(name=name, metrics=metrics, **profile))
                continue

        profile = mock_profiles.get(name, {"bias": {c: 0.0 for c in CLASSES}, "noise": 0.35})
        registry.register(MockBloodDiseaseDetector(name=name, metrics=metrics, **profile))
        print(f"[modelos] '{name}': nenhum arquivo em 'models_sangue/{name}.h5' — usando modelo simulado (mock).")

    return registry
