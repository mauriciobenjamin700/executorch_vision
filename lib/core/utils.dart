/// Biblioteca de utilitários para operações relacionadas à segmentação de imagens.
///
/// Esta biblioteca agrupa funções auxiliares usadas para:
/// - selecionar a melhor detecção de segmentação com base em confiança;
/// - extrair coeficientes de máscara a partir de estruturas de detecção;
/// - construir máscaras binárias a partir de protótipos e coeficientes;
/// - redimensionar e aplicar máscaras a imagens originais;
/// - codificar e decodificar imagens em/para `Uint8List`.
///
/// Uso geral
/// ---------
/// As funções nesta biblioteca são pensadas para um pipeline de segmentação onde:
/// 1. Você obtém várias detecções com scores de confiança e coeficientes de máscara.
/// 2. Seleciona a detecção mais adequada com `getBestSegmentationIndex`.
/// 3. Extrai os coeficientes de máscara com `extractMaskCoefficients`.
/// 4. Constroi uma máscara binária usando `buildBinaryMask`.
/// 5. Se necessário, redimensiona a máscara com `resizeMask` para combinar com a imagem original.
/// 6. Aplica a máscara à imagem original com `applyMaskToImage`.
/// 7. Codifica/decodifica imagens com `encodeImageToPng` / `decodeOriginalImage`.
///
/// Diretrizes de comportamento
/// ---------------------------
/// - Todas as funções devem validar entradas (tamanhos, tipos e limites) e documentar
///   quaisquer exceções lançadas quando a validação falhar.
/// - Operações que envolvem alocação de imagens e operações matriciais devem considerar
///   custo computacional e memória; preferir operações em ponto flutuante quando necessário,
///   e fornecer versões binárias/thresholded para visualização leve.
/// - As máscaras binárias resultantes têm valores booleanos/0-255 e devem ser compatíveis
///   com formatos de imagem comuns (RGBA/RGB).
///
/// Documentação das funções
/// -----------------------
/// getBestSegmentationIndex
///     Retorna o índice da melhor detecção de segmentação com base em um limiar
///     de confiança e, opcionalmente, priorizando detecções por área ou score.
///
///     Parâmetros:
///     - `scores`: lista de probabilidades/score para cada detecção (0.0 - 1.0).
///     - `threshold`: valor mínimo aceitável de confiança para considerar uma detecção.
///     - `preferLargest`: se `true`, desempata escolhendo a detecção de maior área/mascara.
///
///     Retorna:
///     - índice (`int`) da detecção selecionada ou `-1` se nenhuma detecção atingir o limiar.
///
///     Lança:
///     - `ArgumentError` se `scores` for vazio ou `threshold` estiver fora do intervalo válido.
///
/// extractMaskCoefficients
///     Extrai os coeficientes (vetor) de máscara associados a uma detecção específica.
///
///     Parâmetros:
///     - `detection`: estrutura que representa uma detecção (mapa/objeto) contendo campos
///       com coeficientes ou índices para acessar coeficientes.
///     - `index`: índice da detecção a partir da qual extrair os coeficientes.
///
///     Retorna:
///     - `List<double>` com os coeficientes da máscara para a detecção escolhida.
///
///     Observações:
///     - A forma exata de `detection` depende do modelo/formatos upstream; esta função
///       deve documentar qual campo é esperado (por exemplo, `mask_coeffs`, `proto_coef`).
///
/// buildBinaryMask
///     Constrói uma máscara binária (valor booleano ou 0/1) a partir dos protótipos de
///     máscara (`maskPrototypes`) e dos coeficientes extraídos.
///
///     Parâmetros:
///     - `maskPrototypes`: tensor/matriz  HxW x P (protótipos de máscara) ou equivalente.
///     - `maskCoefficients`: vetor de coeficientes correspondente ao proto-dim `P`.
///     - `threshold`: limiar para converter a máscara contínua em binária (padrão recomendado: 0.5).
///
///     Retorna:
///     - Matriz 2D (altura x largura) representando a máscara binária, usando valores
///       0/1 ou `bool`.
///
///     Observações:
///     - Deve normalizar/convertar o resultado linear dos protótipos antes do threshold,
///       tipicamente aplicando `sigmoid` se os protótipos gerarem logits.
///
/// decodeOriginalImage
///     Decodifica bytes de imagem (`Uint8List`) para uma representação manipulável
///     (por exemplo, `image` do pacote `image` ou `ui.Image`), preservando canais e
///     perfil de cores quando possível.
///
///     Parâmetros:
///     - `imageBytes`: bytes da imagem (PNG, JPEG, ...).
///
///     Retorna:
///     - Objeto imagem decodificado apropriado ao ambiente (p.ex. `Image` do pacote `image`
///       ou `ui.Image` para uso em Flutter).
///
///     Lança:
///     - `FormatException` se os bytes não representarem uma imagem válida.
///
/// resizeMask
///     Redimensiona uma máscara (binária ou contínua) para as dimensões alvo,
///     preservando proporções e utilizando interpolação adequada.
///
///     Parâmetros:
///     - `mask`: matriz 2D da máscara (binária ou contínua).
///     - `targetWidth` / `targetHeight`: dimensões alvo em pixels.
///     - `interpolation`: modo de interpolação (ex.: `nearest` para máscaras binárias,
///       `bilinear` para valores contínuos). Deve-se usar `nearest` para preservar bordas quando binária.
///
///     Retorna:
///     - Máscara redimensionada com as dimensões fornecidas.
///
///     Observações:
///     - Para máscaras binárias, após redimensionar com `nearest` garantir que os valores
///       permaneçam 0/1; caso utilize interpolação contínua, aplicar threshold após o resize.
///
/// applyMaskToImage
///     Aplica a máscara binária à imagem original, produzindo uma saída composta que destaca
///     a região segmentada ou aplica transparência fora da máscara.
///
///     Parâmetros:
///     - `image`: imagem original decodificada.
///     - `mask`: máscara binária redimensionada compatível com `image`.
///     - `maskColor` (opcional): cor para preencher a região mascarada ao invés de preservar
///       os pixels originais.
///     - `alpha` (opcional): nível de opacidade aplicado à máscara/overlay.
///
///     Retorna:
///     - Nova imagem com a máscara aplicada (mesmo tipo da entrada).
///
///     Observações:
///     - Deve garantir alinhamento de canais (RGB vs RGBA). Para visualização, é comum
///       aplicar uma leve transparência à máscara para mostrar contexto da imagem original.
///
/// encodeImageToPng
///     Codifica uma imagem em bytes PNG (`Uint8List`).
///
///     Parâmetros:
///     - `image`: representação de imagem a ser codificada.
///     - `quality`/`compression` (opcional): parâmetros de codificação se suportado pelo encoder.
///
///     Retorna:
///     - `Uint8List` contendo os bytes PNG.
///
///     Lança:
///     - Erro se o encoder falhar ou a imagem estiver em formato não suportado.
///
/// getMaxIndexAndProb
///     Obtém o índice e a probabilidade máxima de uma lista de probabilidades/score.
///
///     Parâmetros:
///     - `probs`: lista de probabilidades ou scores (p. ex. saída softmax).
///
///     Retorna:
///     - Tupla/objeto contendo `index` (int) e `prob` (double) correspondentes ao máximo.
///     - Se `probs` estiver vazio, retornar `index = -1` e `prob = 0.0` ou lançar `ArgumentError`
///       conforme a política da biblioteca.
///
/// Exemplos rápidos
/// ----------------
/// - Selecionar a melhor detecção:
///   final idx = getBestSegmentationIndex(scores, threshold: 0.5);
///
/// - Construir e aplicar uma máscara:
///   final coeffs = extractMaskCoefficients(detections[idx], idx);
///   final mask = buildBinaryMask(prototypes, coeffs, threshold: 0.5);
///   final resized = resizeMask(mask, image.width, image.height);
///   final result = applyMaskToImage(image, resized, maskColor: Colors.green.withOpacity(0.4));
///
/// Notas de implementação
/// ---------------------
/// - Preferir operações vetorizadas (matrizes/BLAS) ao combinar protótipos e coeficientes
///   para desempenho em dispositivos móveis/embaracados.
/// - Documentar claramente o formato esperado de `maskPrototypes`, `detection` e quaisquer
///   estruturas intermediárias compartilhadas entre funções para evitar incompatibilidades.
/// - Fornecer testes unitários que validem:
///   * seleção de índice em casos de empate e limiar;
///   * consistência entre construções de máscara contínua e binária;
///   * comportamento de resize com diferentes modos de interpolação;
///   * preservação de canais e transparência em `applyMaskToImage`.
/// Contém funções para processar detecções, construir máscaras binárias e aplicar máscaras a imagens.
///
/// Functions:
/// - getBestSegmentationIndex: Retorna o índice da melhor detecção de segmentação com base em um limiar de confiança.
/// - extractMaskCoefficients: Extrai os coeficientes da máscara para uma detecção específica.
/// - buildBinaryMask: Constrói uma máscara binária a partir dos protótipos de máscara e coeficientes.
/// - decodeOriginalImage: Decodifica uma imagem original a partir de bytes Uint8List.
/// - resizeMask: Redimensiona a máscara para as dimensões alvo.
/// - applyMaskToImage: Aplica a máscara binária à imagem original.
/// - encodeImageToPng: Codifica uma imagem em bytes PNG (Uint8List
/// - getMaxIndexAndProb: Obtém o índice e a probabilidade máxima de uma lista de probabilidades.
library;

import 'dart:math' as math;
import 'dart:typed_data';
import 'package:image/image.dart' as img;

/// Retorna o índice da melhor detecção de segmentação com base em um limiar de confiança.
///
/// Params:
///   - detections: Lista 2D onde cada coluna representa uma detecção e cada linha um atributo (incluindo confiança).
///   - threshold: Limiar mínimo de confiança para considerar uma detecção válida.
///
/// Returns: Índice da melhor detecção ou -1 se nenhuma detecção atender ao limiar.
int getBestSegmentationIndex(List<List<double>> detections, double threshold) {
  double maxConfidence = -1.0;
  int bestDetectionIndex = -1;
  int detectionCount = detections[0].length;

  for (int i = 0; i < detectionCount; i++) {
    final confidence = detections[4][i];
    if (confidence > 0.5 && confidence > maxConfidence) {
      maxConfidence = confidence;
      bestDetectionIndex = i;
    }
  }

  return bestDetectionIndex;
}

/// Extrai os coeficientes da máscara para uma detecção específica.
///
/// Params:
///   - detections: Lista 2D onde cada coluna representa uma detecção e cada  linha um atributo (incluindo coeficientes da máscara).
///   - detectionIndex: Índice da detecção da qual extrair os coeficientes.
///   - firstCoeffIndex: Índice da primeira coluna que contém os coeficientes da máscara.
///
/// Returns: Lista de coeficientes da máscara.
List<double> extractMaskCoefficients(
  List<List<double>> detections,
  int detectionIndex,
  int firstCoeffIndex,
) {
  final maskCoeffs = <double>[];
  for (int i = firstCoeffIndex; i < detections.length; i++) {
    maskCoeffs.add(detections[i][detectionIndex]);
  }
  return maskCoeffs;
}

/// Constrói uma máscara binária a partir dos protótipos de máscara e coeficientes.
///
/// Params:
///   - maskPrototypes: Lista 3D representando os protótipos de máscara (altura x largura x canais).
///   - maskCoefficients: Lista de coeficientes da máscara.
///
/// Returns: Imagem binária resultante.
img.Image buildBinaryMask(
  List<List<List<double>>> maskPrototypes,
  List<double> maskCoefficients,
) {
  final height = maskPrototypes.length;
  final width = maskPrototypes[0].length;
  final channels = maskPrototypes[0][0].length;
  final numCoeffs = math.min(maskCoefficients.length, channels);

  final binaryMask = img.Image(width: width, height: height);

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      double maskValue = 0.0;

      // eixo dos canais no final (y, x, i)
      for (int i = 0; i < numCoeffs; i++) {
        maskValue += maskCoefficients[i] * maskPrototypes[y][x][i];
      }

      // Aplicar função sigmoide
      final sigmoidValue = 1 / (1 + math.exp(-maskValue));

      // Binarizar com threshold 0.5
      final pixelValue = sigmoidValue > 0.5 ? 255 : 0;

      binaryMask.setPixelRgb(x, y, pixelValue, pixelValue, pixelValue);
    }
  }

  return binaryMask;
}

/// Decodifica uma imagem original a partir de bytes Uint8List.
///
/// Params:
///   - bytes: Bytes da imagem original.
///
/// Returns: Imagem decodificada.
img.Image decodeOriginalImage(Uint8List bytes) {
  final image = img.decodeImage(bytes);
  if (image == null) {
    throw Exception("Falha ao decodificar a imagem original.");
  }
  return image;
}

/// Redimensiona a máscara para as dimensões alvo.
///
/// Params:
///   - mask: Imagem da máscara a ser redimensionada.
///  - targetWidth: Largura alvo.
/// - targetHeight: Altura alvo.
///
/// Returns: Imagem da máscara redimensionada.
img.Image resizeMask(img.Image mask, int targetWidth, int targetHeight) {
  return img.copyResize(
    mask,
    width: targetWidth,
    height: targetHeight,
    interpolation: img.Interpolation.linear,
  );
}

/// Aplica a máscara binária à imagem original.
/// Zera os pixels da imagem original onde a máscara é preta.
///
/// Params:
///  - image: Imagem original.
///  - mask: Máscara binária.
///
/// Returns: Imagem resultante com a máscara aplicada.
img.Image applyMaskToImage(img.Image image, img.Image mask) {
  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final maskPixel = mask.getPixel(x, y);
      // Canal R, G e B são iguais, então usamos R
      if (maskPixel.r < 128) {
        image.setPixelRgb(x, y, 0, 0, 0);
      }
    }
  }
  return image;
}

/// Codifica uma imagem em bytes PNG (Uint8List).
///
/// Params:
///   - image: Imagem a ser codificada.
///
/// Returns: Bytes PNG da imagem.
Uint8List encodeImageToPng(img.Image image) {
  return Uint8List.fromList(img.encodePng(image));
}

/// Obtém o índice e a probabilidade máxima de uma lista de probabilidades.
///
/// Params:
///   - probabilities: Lista de probabilidades para cada classe.
///
/// Returns: Tupla contendo o índice da classe com a maior probabilidade e o valor dessa probabilidade.
(int, double) getMaxIndexAndProb(List<double> probabilities) {
  double maxProb = 0.0;
  int maxIndex = -1;
  for (int i = 0; i < probabilities.length; i++) {
    if (probabilities[i] > maxProb) {
      maxProb = probabilities[i];
      maxIndex = i;
    }
  }
  if (maxIndex == -1) {
    throw Exception('No classification result found.');
  }

  return (maxIndex, maxProb);
}
