/// Biblioteca principal do pacote `executorch_vision`.
///
/// Esta biblioteca reexporta os módulos centrais do pacote para proporcionar
/// uma API única e de fácil utilização para tarefas comuns de visão por
/// computador com modelos TFLite/Executorch. Fornece ferramentas para
/// pré-processamento de imagens, definição e carregamento de modelos,
/// estruturas de resultados de inferência (schemas) e utilitários auxiliares
/// para manipulação de imagens e dados.
///
/// Principais objetivos
/// - Simplificar o fluxo de inferência: carregar modelo → pré-processar
///   imagem → executar inferência → interpretar resultados.
/// - Fornecer tipos e contratos bem documentados (schemas) para resultados,
///   facilitando integração com UI e pipelines assíncronos.
/// - Reexportar utilitários comuns para evitar imports fragmentados.
///
/// Módulos reexportados
/// - `pre_processing.dart`
///   - Funções e classes para preparar imagens antes da inferência:
///     redimensionamento, centralização/crop, conversão de formato (RGB/BGR),
///     normalização, conversão para tensores/arrays compatíveis com TFLite.
///   - Projeta-se para ser composável: você pode encadear transformações
///     leves e reutilizáveis.
/// - `models.dart`
///   - Definições de modelos suportados, helpers para carregar modelos
///     TFLite/Executorch e gerenciar recursos (asset vs arquivo local).
///   - Abstrações para encapsular metadados do modelo (input shape, tipos,
///     rótulos, quantização).
/// - `schemas.dart`
///   - Classes de domínio que representam resultados de inferência:
///     detecção de objetos, classificação, segmentação, keypoints, etc.
///   - Métodos utilitários para serialização/deserialização e mapeamento
///     entre coordenadas de imagem e coordenadas normalizadas.
/// - `utils.dart`
///   - Utilitários auxiliares: conversões de imagem↔bytes, manipulação de
///     buffers, operações de performance (pooling/reuso de buffers),
///     ajuda para operações assíncronas e validação de entrada.
///
/// Uso rápido
/// ```dart
/// import 'package:executorch_vision/executorch_vision.dart';
///
/// // 1. Carregue um modelo (ex.: classe do arquivo models.dart)
/// // 2. Pré-processe a imagem (ex.: redimensionar, normalizar)
/// // 3. Execute a inferência e obtenha um schema com resultados
/// // 4. Interprete e apresente os resultados na UI
/// ```
///
/// Exemplo (fluxo típico)
/// ```dart
/// // Carrega modelo
/// final model = await ModelLoader.loadFromAsset('assets/model.tflite');
///
/// // Pré-processamento
/// final input = await PreProcessing.resizeAndNormalize(image,
///   width: model.inputWidth, height: model.inputHeight);
///
/// // Inferência
/// final rawOutput = await model.runInference(input);
///
/// // Conversão para schema de alto nível
/// final results = InferenceSchema.fromRaw(model, rawOutput);
///
/// // Uso dos resultados (ex.: desenhar bounding boxes)
/// for (final detection in results.detections) {
///   // mapear coordenadas normalizadas para coordenadas da imagem
///   final rect = detection.mapToImageRect(image.width, image.height);
///   // renderizar rect na UI...
/// }
/// ```
///
/// Boas práticas
/// - Reutilize buffers/tensores quando possível para reduzir transferência de
///   memória e GC. Os utilitários em `utils.dart` facilitam esse reuso.
/// - Conheça a forma de entrada do seu modelo (shape, ordenação de canais,
///   normalização). As abstrações em `models.dart` carregam metadados que
///   ajudam a evitar erros de shape/escala.
/// - Execute operações pesadas (pre-processamento grande, pós-processamento
///   intensivo) fora da UI thread, usando isolates ou APIs assíncronas.
///
/// Confiabilidade e notas de implementação
/// - Projetado para ser compatível com null-safety do Dart.
/// - Tratamento de erros: as funções públicas lançam exceções claras em caso
///   de entrada inválida, modelo incompatível ou falha ao carregar recursos.
/// - Plataformas suportadas: mobile (Android/iOS) e desktop onde TFLite/Backends
///   são compatíveis; confira a documentação do backend de inferência escolhido.
///
/// Exemplos e testes
/// - Consulte a pasta `example/` do pacote para exemplos completos de integração
///   com Flutter e pipelines de inferência em tempo real.
/// - Tests unitários e de integração cobrem transformações de
///   pré-processamento e parsing de schemas.
///
/// Ver também
/// - `pre_processing.dart` — transformações de imagem e utilitários de entrada.
/// - `models.dart` — carregamento e metadados de modelos.
/// - `schemas.dart` — representação tipada dos resultados de inferência.
/// - `utils.dart` — funções auxiliares de conversão e performance.
///
/// Licença e contribuição
/// - Veja o arquivo LICENSE na raiz do repositório para detalhes de licença.
/// - Contribuições são bem-vindas; abra issues/PRs com descrições claras do
///   problema ou feature desejada.
/// lib/executorch_vision.dart
/// Biblioteca principal para o pacote `executorch_vision`.
/// Reexporta módulos centrais para fácil acesso.
///
/// Módulos Reexportados:
/// - pre_processing.dart: Pré-processamento de imagens.
/// - models.dart: Definição de modelos TFLite suportados.
/// - schemas.dart: Definição de classes de resultados de inferência.
/// - utils.dart: Utilitários auxiliares para manipulação de imagens e dados.
library;

export 'core/pre_processing.dart';
export 'core/models.dart';
export 'core/schemas.dart';
export 'core/utils.dart';
