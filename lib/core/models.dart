/// Biblioteca de utilitários para carregar e gerenciar modelos de visão computacional
/// usando ExecuTorch.
///
/// Esta biblioteca fornece abstrações e contratos claros para trabalhar com três
/// perfis de modelos comuns: detecção de objetos, segmentação de imagens e
/// classificação de imagens. Apresenta propriedades reutilizáveis (caminho do
/// modelo, dimensões de entrada, rótulos, parâmetros de pré/pós-processamento)
/// e descreve o formato esperado de entrada/saída para facilitar integração e testes.
///
/// Uso (exemplo ilustrativo)
/// ```dart
/// // Exemplo ilustrativo — adapte à API concreta do ExecuTorch no seu projeto.
/// final model = DetectionModel(modelPath: 'assets/models/detector.pt', inputSize: Size(640,640));
/// final results = await model.predict(image); // retornará lista de detecções
/// ```
///
/// Classes fornecidas
/// - DetectionModel
///   - Propósito: detectar objetos em imagens e retornar bounding boxes, scores e rótulos.
///   - Entrada: imagem (ex.: Uint8List, Image, ou outra representação suportada),
///     dimensões de entrada (width × height) para redimensionamento/preproc.
///   - Saída prevista: lista de detecções com {box: Rect, score: double, label: String, classId: int}.
///   - Configurações típicas: threshold de confiança, NMS (non-maximum suppression),
///     limite máximo de detecções, formato de coordenadas (normalizado ou pixels).
///
/// - SegmentModel
///   - Propósito: gerar máscaras de segmentação por objeto ou por classe para uma imagem.
///   - Saída prevista: máscaras binárias ou mapas de probabilidade (por-pixel), opcionalmente
///     associadas a rótulos e scores; formatos retornados podem incluir arrays de floats,
///     bitmaps ou imagens RGBA conforme configuração.
///   - Observações: cuidado com alinhamento entre máscara e imagem original (resize / pad).
///
/// - ClassifyModel
///   - Propósito: classificar imagens e retornar uma lista ordenada de rótulos com scores.
///   - Saída prevista: lista de {label: String, score: double, classId: int}, tipicamente
///     ordenada por probabilidade decrescente.
///   - Observações: suporte a top-k, normalização de scores (softmax) e transformação de entrada.
///
/// Propriedades comuns e contratos
/// - modelPath: caminho para o arquivo do modelo (assets, pacote, caminho absoluto).
/// - inputSize: dimensões (width, height) esperadas pelo modelo; o loader deve aplicar
///   redimensionamento e, se necessário, padding para manter razão de aspecto.
/// - labelsPath: arquivo de rótulos (um por linha). Se ausente, as classes ficam indexadas por inteiro.
/// - normalization (mean / std): parâmetros para normalizar pixels antes da inferência.
/// - batchSize: quando suportado pelo backend, permita inferência em lote.
/// - Formatos de modelo: dependem do backend ExecuTorch utilizado (p. ex. TorchScript, ONNX, TFLite).
///
/// Comportamento, erros e garantias
/// - Validação: carregadores devem lançar exceções claras para erros como arquivo não encontrado,
///   formato inválido ou incompatibilidade de dimensão.
/// - Performance: carregamento de modelo é custoso — reutilize instâncias carregadas e evite
///   recarregar em loop; ofereça inicialização assíncrona (async) no app.
/// - Thread-safety: documente se instâncias são seguras para uso concorrente; caso contrário,
///   exponha factories/locks para gerenciar acesso.
/// - Precisão dos resultados: resultados pós-processados (NMS, threshold, mapeamento de rótulos)
///   devem ser reproduzíveis e configuráveis via parâmetros públicos.
///
/// Boas práticas recomendadas
/// - Documente explicitamente formatos de entrada (RGB vs BGR, ordem de canais, range 0..1 vs 0..255).
/// - Forneça utilitários de pré/processamento reutilizáveis (resize + pad + normalize).
/// - Exponha parâmetros de pós-processamento (NMS, score threshold, top-k) na API pública.
/// - Inclua exemplos e testes de inferência com modelos toy para garantir compatibilidade entre versões.
///
/// Exemplo de contrato de saída (detecção)
/// ```dart
/// /// Representação de uma detecção retornada por DetectionModel
/// class Detection {
///   final Rect box;       // Caixa em coordenadas da imagem original
///   final double score;   // Confiança (0.0 - 1.0)
///   final String label;   // Rótulo legível
///   final int classId;    // ID numérico da classe
/// }
/// ```
///
/// Observações finais
/// - Documente no README do pacote procedimentos de conversão de modelos, formatos suportados,
///   e exemplos completos de pré/pós-processamento aplicáveis ao pipeline ExecuTorch.
/// - Sempre forneça testes de integração que executem inferência end-to-end com pequenos modelos
///   para detectar regressões de I/O, dimensionamento e mapeamento de rótulos.
/// Biblioteca para carregar e gerenciar modelos de visão computacional usando ExecuTorch.
/// Define classes para modelos de detecção, segmentação e classificação.
/// Propriedades comuns incluem caminho do modelo, dimensões de entrada e rótulos.
/// Módulos Definidos:
///   - DetectionModel: Classe para modelos de detecção de objetos.
///   - SegmentModel: Classe para modelos de segmentação de imagens.
///   - ClassifyModel: Classe para modelos de classificação de imagens.
library;

import 'dart:convert';
import 'dart:developer';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import './pre_processing.dart';
import './schemas.dart';
import './utils.dart';

Future<ExecuTorchModel> loadModel(String modelPath) async {
  final byteData = await rootBundle.load(modelPath);
  final tempDir = await getTemporaryDirectory();
  final modelTempName = modelPath.split('/').last;
  // Avoid appending an extra .pte if modelPath already contains the extension
  final file = File('${tempDir.path}/$modelTempName');
  await file.writeAsBytes(byteData.buffer.asUint8List());
  log('Loading model from: ${file.path}');
  try {
    final model = await ExecuTorchModel.load(file.path);
    log('ExecuTorchModel.load returned: $model');
    return model;
  } catch (e) {
    log('Erro ao carregar ExecuTorchModel: $e');
    rethrow;
  }
}

Future<List<String>> loadLabels(String labelsPath) async {
  final labelsData = await rootBundle.loadString(labelsPath);
  final labels = const LineSplitter().convert(labelsData);
  log('Rótulos carregados de $labelsPath');
  return labels;
}

class DetectionModel {
  final ExecuTorchModel model;
  final String modelPath;
  final int inputWidth;
  final int inputHeight;
  final List<String> labels;

  DetectionModel({
    required this.model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });

  Future<List<DetectionResult>> predict(
    Uint8List inputData, {
    double confThreshold = 0.25,
  }) async {
    final tensorData = await PreProcessing.toTensorData(
      inputData,
      targetWidth: inputWidth,
      targetHeight: inputHeight,
    );

    // Executa a inferência
    final outputs = await model.forward([tensorData]);

    log('Inferência concluída no modelo de detecção');
    for (var output in outputs) {
      log('Output shape: ${output.shape}');
      log('Output type: ${output.dataType}');
      log('Output data length: ${output.data.length}');
      log('Output data (first 10 values): ${output.data.take(10).toList()}');
    }
    if (outputs.isEmpty) return <DetectionResult>[];

    final out = outputs[0];
    // Expecting shape like [1, channels, num_boxes]
    final shape = out.shape;
    if (shape.length < 3) return <DetectionResult>[];
    final channels = shape[1]!;
    final numBoxes = shape[2]!;

    // Convert bytes to float32 list (respecting offset)
    final floatData = out.data.buffer.asFloat32List(
      out.data.offsetInBytes,
      out.data.lengthInBytes ~/ 4,
    );
    log('Parsed float length: ${floatData.length}');
    try {
      log('First 10 floats: ${floatData.take(10).toList()}');
    } catch (e) {
      log('Could not print float sample: $e');
    }

    // Helpers
    double sigmoid(double x) => 1.0 / (1.0 + math.exp(-x));
    final List<DetectionResult> detections = [];

    final expectedWithObj = labels.length + 5;
    final hasObjectness = channels == expectedWithObj;
    final classCount = hasObjectness ? (channels - 5) : (channels - 4);
    log(
      'channels=$channels numBoxes=$numBoxes expectedWithObj=$expectedWithObj hasObjectness=$hasObjectness classCount=$classCount',
    );

    for (var i = 0; i < numBoxes; i++) {
      // index by channel-first layout: index = c * numBoxes + i
      double at(int c) => floatData[c * numBoxes + i];

      final x = at(0);
      final y = at(1);
      final w = at(2);
      final h = at(3);

      double objectness = 1.0;
      int classOffset = 4;
      if (hasObjectness) {
        objectness = sigmoid(at(4));
        classOffset = 5;
      }

      final classScores = List<double>.generate(
        classCount,
        (j) => at(classOffset + j),
      );
      if (i < 3)
        log(
          'box $i raw x,y,w,h: $x,$y,$w,$h objectness:$objectness'
          ' classScoresSample:${classScores.take(6).toList()}',
        );
      final classProbs = classScores.map((s) => sigmoid(s)).toList();
      double maxClassProb = classProbs.reduce(math.max);
      final classId = classProbs.indexWhere((p) => p == maxClassProb);

      final conf = hasObjectness ? (objectness * maxClassProb) : maxClassProb;
      if (conf < confThreshold) continue;

      // Assume x,y,w,h are normalized center coords (0..1)
      final cx = x;
      final cy = y;
      final bw = w;
      final bh = h;

      double left = (cx - bw / 2.0) * inputWidth;
      double top = (cy - bh / 2.0) * inputHeight;
      double right = (cx + bw / 2.0) * inputWidth;
      double bottom = (cy + bh / 2.0) * inputHeight;

      // Clamp
      left = left.clamp(0.0, inputWidth.toDouble());
      top = top.clamp(0.0, inputHeight.toDouble());
      right = right.clamp(0.0, inputWidth.toDouble());
      bottom = bottom.clamp(0.0, inputHeight.toDouble());

      final label = (classId >= 0 && classId < labels.length)
          ? labels[classId]
          : 'class_$classId';

      detections.add(
        DetectionResult(
          classId: classId,
          label: label,
          confidence: conf,
          bbox: [left, top, right, bottom],
        ),
      );
    }

    // Sort by confidence descending
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    return detections;
  }
}

///////////// SEG

class SegmentModel {
  final ExecuTorchModel model;
  final String modelPath;
  final int inputWidth;
  final int inputHeight;
  final List<String> labels;

  SegmentModel({
    required this.model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });

  Future<SegmentationResult> predict(Uint8List originalImageBytes) async {
    final inputTensor = await PreProcessing.toTensorData(
      originalImageBytes,
      targetWidth: inputWidth,
      targetHeight: inputHeight,
    );
    final outputs = await forward(inputTensor, originalImageBytes);
    return getResult(outputs, originalImageBytes);
  }

  Future<List<TensorData>> forward(
    TensorData inputTensor,
    Uint8List originalImageBytes,
  ) async {
    try {
      final outputs = await model.forward([inputTensor]);

      return outputs;
    } catch (error) {
      rethrow;
    }
  }

  SegmentationResult getResult(
    List<TensorData> outputs,
    Uint8List originalImageBytes,
  ) {
    final segmentationThreshold = 0.5; // Threshold de confiança
    final firstCoeffIndex =
        5; // Índice da primeira coluna dos coeficientes da máscara
    double confidence;
    // Converte outputs[0] -> segmentations (esperado como [1, channels, num_detections])
    final outSeg = outputs[0];
    // Ler floats de forma segura mesmo que o Uint8List seja um view desalinhado
    final bdSeg = ByteData.sublistView(outSeg.data);
    final floatLenSeg = bdSeg.lengthInBytes ~/ 4;
    final floatSeg = Float32List(floatLenSeg);
    for (var i = 0; i < floatLenSeg; i++) {
      floatSeg[i] = bdSeg.getFloat32(i * 4, Endian.little);
    }
    final shapeSeg = outSeg.shape;
    final channels = shapeSeg.length >= 3 ? shapeSeg[1]! : 0;
    final numDet = shapeSeg.length >= 3 ? shapeSeg[2]! : 0;

    final segmentations = List<List<double>>.generate(
      channels,
      (_) => List<double>.filled(numDet, 0.0),
    );
    for (int c = 0; c < channels; c++) {
      for (int i = 0; i < numDet; i++) {
        segmentations[c][i] = floatSeg[c * numDet + i];
      }
    }

    // Converte outputs[1] -> mask prototypes (pode estar em NCHW ou NHWC)
    final outProto = outputs[1];
    final bdProto = ByteData.sublistView(outProto.data);
    final floatLenProto = bdProto.lengthInBytes ~/ 4;
    final floatProto = Float32List(floatLenProto);
    for (var i = 0; i < floatLenProto; i++) {
      floatProto[i] = bdProto.getFloat32(i * 4, Endian.little);
    }
    final shapeProto = outProto.shape;

    late final List<List<List<double>>> maskPrototypes;
    if (shapeProto.length >= 4) {
      final a = shapeProto[1]!;
      final b = shapeProto[2]!;
      final d = shapeProto[3]!;
      // Detecta se o layout é NCHW ([1, C, H, W]) ou NHWC ([1, H, W, C])
      final floatLen = floatProto.length;
      final maybeC = floatLen ~/ (b * d);
      if (maybeC == a) {
        // NCHW -> converter para [H][W][C]
        final C = a;
        final H = b;
        final W = d;
        maskPrototypes = List.generate(
          H,
          (_) => List.generate(W, (_) => List<double>.filled(C, 0.0)),
        );
        for (int c = 0; c < C; c++) {
          final base = c * H * W;
          for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
              maskPrototypes[y][x][c] = floatProto[base + y * W + x];
            }
          }
        }
      } else {
        // NHWC -> [1, H, W, C]
        final H = a;
        final W = b;
        final C = d;
        maskPrototypes = List.generate(
          H,
          (_) => List.generate(W, (_) => List<double>.filled(C, 0.0)),
        );
        for (int y = 0; y < H; y++) {
          for (int x = 0; x < W; x++) {
            final base = (y * W + x) * C;
            for (int c = 0; c < C; c++) {
              maskPrototypes[y][x][c] = floatProto[base + c];
            }
          }
        }
      }
    } else {
      throw Exception('Formato inesperado para protótipos de máscara');
    }

    final bestSegmentationIndex = getBestSegmentationIndex(
      segmentations,
      segmentationThreshold,
    );
    if (bestSegmentationIndex == -1) {
      throw Exception("No found segmentations.");
    }

    final maskCoeffs = extractMaskCoefficients(
      segmentations,
      bestSegmentationIndex,
      firstCoeffIndex,
    );

    final binaryMask = buildBinaryMask(maskPrototypes, maskCoeffs);
    final originalImage = decodeOriginalImage(originalImageBytes);
    final resizedMask = resizeMask(
      binaryMask,
      originalImage.width,
      originalImage.height,
    );
    final maskedImage = applyMaskToImage(originalImage, resizedMask);
    final segmentedImageBytes = encodeImageToPng(maskedImage);

    confidence = bestSegmentationIndex < segmentations[0].length
        ? segmentations[4][bestSegmentationIndex]
        : 0.0;

    final segLabel =
        (bestSegmentationIndex >= 0 && bestSegmentationIndex < labels.length)
        ? labels[bestSegmentationIndex]
        : null;

    return SegmentationResult(
      originalImage: originalImageBytes,
      segmentedImage: segmentedImageBytes,
      binaryMask: binaryMask,
      maskCoefficients: maskCoeffs,
      bestSegmentationIndex: bestSegmentationIndex,
      confidence: confidence,
      timestamp: DateTime.now(),
      metadata: {
        'threshold': segmentationThreshold,
        'firstCoeffIndex': firstCoeffIndex,
        'originalImageSize': '${originalImage.width}x${originalImage.height}',
        'label': segLabel,
      },
    );
  }
}

///////////// CLASSIFY
///
///

class ClassifyModel {
  final ExecuTorchModel model;
  final String modelPath;
  final int inputWidth;
  final int inputHeight;
  final List<String> labels;

  ClassifyModel({
    required this.model,
    required this.modelPath,
    required this.inputWidth,
    required this.inputHeight,
    required this.labels,
  });

  /// Realiza a predição de classificação em uma imagem de entrada.
  /// Retorna um [ClassificationResult] contendo os resultados da classificação.
  Future<ClassificationResult> predict(Uint8List originalImageBytes) async {
    final inputTensor = await PreProcessing.toTensorData(
      originalImageBytes,
      targetWidth: inputWidth,
      targetHeight: inputHeight,
    );
    final outputs = await forward(inputTensor, originalImageBytes);
    return getResult(outputs, originalImageBytes);
  }

  /// Executa a inferência do modelo com o tensor de entrada.
  /// Retorna um mapa contendo as saídas do modelo.
  Future<List<TensorData>> forward(
    TensorData inputTensor,
    Uint8List originalImageBytes,
  ) async {
    try {
      final output = await model.forward([inputTensor]);
      return output;
    } catch (error) {
      rethrow;
    }
  }

  /// Processa as saídas do modelo para gerar o resultado de classificação
  /// Retorna um [ClassificationResult] contendo os resultados da classificação.
  ClassificationResult getResult(
    List<TensorData> outputs,
    Uint8List originalImageBytes,
  ) {
    final out = outputs[0];
    // Converte o buffer de bytes do tensor para Float32List de forma segura
    final bd = ByteData.sublistView(out.data);
    final floatCount = bd.lengthInBytes ~/ 4;
    final floatData = Float32List(floatCount);
    for (var i = 0; i < floatCount; i++) {
      floatData[i] = bd.getFloat32(i * 4, Endian.little);
    }
    // Cria uma lista de probabilidades em double
    final probabilities = floatData.map((e) => e.toDouble()).toList();
    final (maxIndex, maxProb) = getMaxIndexAndProb(probabilities);

    return ClassificationResult(
      originalImage: originalImageBytes,
      label: labels[maxIndex],
      confidence: maxProb,
      allProbabilities: probabilities,
      timestamp: DateTime.now(),
    );
  }
}
