/// Biblioteca de pré-processamento de imagens para modelos de visão computacional.
///
/// Fornece utilitários para:
/// - converter entre ui.Image e bytes (PNG),
/// - redimensionar e normalizar imagens para entrada de modelos (TensorData),
/// - reconstruir imagens a partir de TensorData.
///
/// A classe PreProcessing assume o layout de tensor NCHW ([1, 3, H, W]) com dtype float32
/// e normalização ImageNet (mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]).
/// Os canais estão em ordem RGB e os valores de pixel são normalizados para a faixa 0–1
/// antes da aplicação da normalização ImageNet.
///
/// Exemplo de uso:
/// ```dart
/// // Converter ui.Image para TensorData pronto para inferência:
/// final bytes = await PreProcessing.fromUiImage(myUiImage);
/// final tensor = await PreProcessing.toTensorData(bytes, targetWidth: 640, targetHeight: 640);
///
/// // Reconstruir PNG a partir de TensorData (desfaz a normalização ImageNet):
/// final pngBytes = await PreProcessing.fromTensorData(tensor);
/// final uiImage = await PreProcessing.toUiImage(pngBytes);
/// ```
///
/// Comportamento e erros:
/// - fromUiImage: retorna Uint8List em formato PNG; lança Exception se a conversão falhar.
/// - toUiImage: decodifica bytes em ui.Image; lança Exception se a decodificação falhar.
/// - toTensorData:
///   - redimensiona a imagem para [targetWidth, targetHeight],
///   - normaliza usando ImageNet e produz TensorData com shape [1, 3, H, W] e TensorType.float32,
///   - lança Exception se a imagem não puder ser decodificada ou se o tamanho dos dados for incompatível.
/// - fromTensorData: reconstrói PNG a partir de TensorData assumindo layout [1,3,H,W] e normalização ImageNet;
///   lança Exception se o formato do tensor for inconsistente.
///
/// Observações de integração:
/// - Ajuste targetWidth/targetHeight conforme exigido pelo seu modelo.
/// - Se o modelo usar BGR, escala 0–255 diretamente ou outro layout (NHWC), adapte as funções de pré-processamento.
library;

import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:image/image.dart' as img;

class PreProcessing {
  /// Converte um [ui.Image] em bytes PNG ([Uint8List]).
  static Future<Uint8List> fromUiImage(ui.Image image) async {
    final byteData = await image.toByteData(format: ui.ImageByteFormat.png);
    if (byteData == null) {
      throw Exception('Falha ao converter imagem para bytes');
    }
    return byteData.buffer.asUint8List();
  }

  static Future<ui.Image> toUiImage(Uint8List bytes) async {
    final codec = await ui.instantiateImageCodec(bytes);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  static Future<TensorData> toTensorData(
    Uint8List imageBytes, {
    int targetWidth = 640,
    int targetHeight = 640,
  }) async {
    final image = img.decodeImage(imageBytes);

    if (image == null) {
      throw Exception('Falha ao decodificar imagem');
    }

    // Redimensionar para 640x640 (tamanho esperado pelo YOLO11n)
    final resized = img.copyResize(
      image,
      width: targetWidth,
      height: targetHeight,
    );
    // Normalizar os pixels (0-1) e aplicar normalização ImageNet
    final normalizedData = <double>[];
    // Valores de média e desvio padrão para ImageNet
    final mean = [0.485, 0.456, 0.406];
    // Valores de desvio padrão para ImageNet
    final std = [0.229, 0.224, 0.225];

    // Percorre os pixels da imagem redimensionada
    for (int i = 0; i < resized.height; i++) {
      // Percorre cada coluna
      for (int j = 0; j < resized.width; j++) {
        // Obtém o pixel na posição (j, i)
        final pixel = resized.getPixelSafe(j, i);

        // Extrair RGB (valores de 0 a 255)
        final r = pixel.rNormalized.toDouble();
        final g = pixel.gNormalized.toDouble();
        final b = pixel.bNormalized.toDouble();

        // Normalizar (ImageNet)
        normalizedData.add((r - mean[0]) / std[0]); // R
        normalizedData.add((g - mean[1]) / std[1]); // G
        normalizedData.add((b - mean[2]) / std[2]); // B
      }
    }

    // Converter para Float32List
    final float32Data = Float32List.fromList(normalizedData);
    // Obter bytes do buffer (cada float -> 4 bytes)
    final resizedImage = float32Data.buffer.asUint8List();

    // Número de elementos esperados para shape [1,3,H,W]
    final expectedElements = 1 * 3 * targetHeight * targetWidth;
    if (float32Data.length != expectedElements) {
      throw Exception(
        'Tamanho dos dados incompatível: esperado $expectedElements, encontrado ${float32Data.length}',
      );
    }

    // Converte os bytes para TensorData usando shape dinâmico [N, C, H, W]
    final inputTensor = TensorData(
      shape: [1, 3, targetHeight, targetWidth],
      dataType: TensorType.float32,
      data: resizedImage,
    );

    return inputTensor;
  }

  static Future<Uint8List> fromTensorData(TensorData tensorData) async {
    // Converter os dados do tensor de volta para Float32List
    final floatData = Float32List.view(tensorData.data.buffer);
    // Obter as dimensões da imagem
    final int width = tensorData.shape[3] as int;
    // Obter as dimensões da imagem
    final int height = tensorData.shape[2] as int;

    // Criar uma imagem vazia
    final image = img.Image(width: width, height: height);

    // Preencher a imagem com os dados do tensor
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        final r = ((floatData[(i * width + j) * 3] * 0.229 + 0.485) * 255)
            .clamp(0, 255)
            .toInt();
        final g = ((floatData[(i * width + j) * 3 + 1] * 0.224 + 0.456) * 255)
            .clamp(0, 255)
            .toInt();
        final b = ((floatData[(i * width + j) * 3 + 2] * 0.225 + 0.406) * 255)
            .clamp(0, 255)
            .toInt();

        image.setPixelRgb(j, i, r, g, b);
      }
    }
    // Codificar a imagem de volta para PNG
    return Uint8List.fromList(img.encodePng(image));
  }
}
