class Report {
  final String date;
  final String score;
  final String origB64;
  final String pseudoB64;
  final List<Map<String, dynamic>>? gradcamLayers;
  final String? primaryGradcamB64;
  final Map<String, dynamic>? probs;
  final num? confidence;
  final String? fileName;
  final String? imageSize;
  final String? probs_chart_b64;
  final bool? isHneCheckbox;
  final String? generated_b64; 

  Report({
    required this.date,
    required this.score,
    required this.origB64,
    required this.pseudoB64,
    this.gradcamLayers,
    this.primaryGradcamB64,
    this.probs_chart_b64,
    this.probs,
    this.confidence,
    this.fileName,
    this.imageSize,
    this.isHneCheckbox,
    this.generated_b64,
  });

  Map<String, dynamic> toJson() => {
    'date': date,
    'score': score,
    'origB64': origB64,
    'pseudoB64': pseudoB64,
    'gradcamLayers': gradcamLayers,
    'primaryGradcamB64': primaryGradcamB64,
    'probs': probs,
    'confidence': confidence,
    'fileName': fileName,
    'imageSize': imageSize,
    'probs_chart_b64' : probs_chart_b64,
    'isHneCheckbox': isHneCheckbox,
    'generated_b64': generated_b64,
  };

  factory Report.fromJson(Map<String, dynamic> json) {
    return Report(
      date: (json['date'] ?? '').toString(),
      score: (json['score'] ?? '').toString(),
      origB64: (json['origB64'] ?? '').toString(),
      generated_b64: (json['generated_b64'] ?? '').toString(),
      pseudoB64: (json['pseudoB64'] ?? '').toString(),
      gradcamLayers: json['gradcamLayers'] != null ? List<Map<String, dynamic>>.from(json['gradcamLayers'].map((e) => Map<String, dynamic>.from(e))) : null,
      primaryGradcamB64: json['primaryGradcamB64']?.toString(),
      probs: json['probs'] != null ? Map<String, dynamic>.from(json['probs']) : null,
      confidence: json['confidence'] != null ? num.tryParse(json['confidence'].toString()) : null,
      fileName: json['fileName']?.toString(),
      probs_chart_b64: (json['probs_chart_b64'] ?? '').toString(),
      imageSize: json['imageSize']?.toString(),
      isHneCheckbox: json['isHneCheckbox'] is bool ? json['isHneCheckbox'] : (json['isHneCheckbox'] == 'true'),
    );
  }
}
