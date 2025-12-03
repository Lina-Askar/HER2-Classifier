import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:pdf/pdf.dart';
import 'package:printing/printing.dart';
import 'dart:convert';
import 'package:flutter/services.dart';
import 'Report.dart';
import 'Result.dart';
import '../widgets/sidebar.dart';
import '../theme_provider.dart';
  Future<void> printSelectedReports(List<Report> selectedReports) async {
    final pdf = pw.Document();
    final prefs = await SharedPreferences.getInstance();
    final String currentUsername = prefs.getString('current_username') ?? '-';
    for (final report in selectedReports) {
      String slideName = (report.fileName ?? '-').split('/').last;
      String username = report.toJson().containsKey('username') && report.toJson()['username'] != null && report.toJson()['username'] != '-' && (report.toJson()['username'] as String).isNotEmpty
          ? report.toJson()['username']
          : currentUsername;
      final String imageType = report.isHneCheckbox == true ? 'H&E (converted to IHC)' : 'IHC';
      String imageSize = (report.imageSize ?? '-');
      if (imageSize != '-' && imageSize.contains('x')) {
        imageSize = imageSize.replaceAll(' ', '');
        imageSize = imageSize.replaceAll('×', 'x');
        var parts = imageSize.split('x');
        if (parts.length == 2) {
          imageSize = '${parts[0]} × ${parts[1]} px';
        }
      }
      final String dateTime = report.date.replaceAll('T', ' ');
      final String analyst = 'HER2 Classifier AI Model';
      final List<String> probKeys = ['0', '1+', '2+', '3+'];
      final Map<String, double> probabilities = {};
      if (report.probs != null) {
        for (var k in probKeys) {
          var v = report.probs![k];
          if (v is num) probabilities[k] = v.toDouble();
        }
      }
      double? confidence = report.confidence?.toDouble();
      if (confidence == null && probabilities.isNotEmpty) {
        confidence = probabilities.values.reduce((a, b) => a > b ? a : b);
      }
      String formatPercent(num? v) => v == null ? '—' : '${(v * 100).toStringAsFixed(2)}%';
      Uint8List? inputImageBytes;
      if (report.origB64.isNotEmpty) {
        inputImageBytes = base64Decode(report.origB64);
      }
      Uint8List? generatedImageBytes;
      if (report.generated_b64 != null && report.generated_b64!.isNotEmpty) {
        generatedImageBytes = base64Decode(report.generated_b64!);
      }
      final List<Uint8List> pseudoImages = [];
      if (report.pseudoB64.isNotEmpty) {
        pseudoImages.add(base64Decode(report.pseudoB64));
      }
      final List<Uint8List> gradCamImages = [];
      if (report.gradcamLayers != null && report.gradcamLayers!.isNotEmpty) {
        for (var layer in report.gradcamLayers!) {
          if (layer['gradcam_b64'] != null && (layer['gradcam_b64'] as String).isNotEmpty) {
            gradCamImages.add(base64Decode(layer['gradcam_b64']));
          }
        }
      }
      String interpretation = '';
      switch (report.score) {
        case '0':
          interpretation = 'The AI system detected low HER2 expression corresponding to a score of 0. This indicates minimal or no membrane staining, consistent with a HER2-negative phenotype.';
          break;
        case '1+':
          interpretation = 'The AI system indicates HER2 Score 1+ (negative). Faint/incomplete membrane staining is observed in tumor cells.';
          break;
        case '2+':
          interpretation = 'The AI system indicates HER2 Score 2+ (equivocal). Weak to moderate complete membrane staining may be present. Additional confirmatory assessment is recommended to determine HER2 amplification status.';
          break;
        case '3+':
          interpretation = 'The AI system indicates HER2 Score 3+ (positive). Intense, complete membrane staining is identified in tumor cells.';
          break;
        default:
          interpretation = '—';
      }
      final ByteData logoData = await rootBundle.load('lib/assets/images/logo.png');
      final Uint8List logoBytes = logoData.buffer.asUint8List();
      pdf.addPage(
        pw.Page(
          margin: const pw.EdgeInsets.all(40),
          build: (pw.Context context) {
            return pw.Column(
              crossAxisAlignment: pw.CrossAxisAlignment.start,
              children: [
                pw.Center(
                  child: pw.Column(
                    children: [
                      pw.Image(
                        pw.MemoryImage(logoBytes),
                        height: 32,
                      ),
                      pw.SizedBox(height: 8),
                      pw.Text(
                        'HER2 Analysis Report',
                        style: pw.TextStyle(
                          fontSize: 18,
                          fontWeight: pw.FontWeight.bold,
                          color: PdfColor.fromHex('#204080'),
                        ),
                      ),
                    ],
                  ),
                ),
                pw.SizedBox(height: 8),
                pw.Center(
                  child: pw.Text(
                    'Date: $dateTime | Uploaded By: $username | Analyst: $analyst',
                    style: pw.TextStyle(fontSize: 9),
                  ),
                ),
                pw.SizedBox(height: 18),
                pw.Text('1. Sample Information', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
                pw.SizedBox(height: 6),
                pw.Table(
                  border: pw.TableBorder.all(color: PdfColor.fromHex('#204080'), width: 1.2),
                  columnWidths: {
                    0: const pw.FlexColumnWidth(1),
                    1: const pw.FlexColumnWidth(1),
                  },
                  children: [
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('File Name', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(slideName, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Image Type', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(imageType, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Image Size', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(imageSize, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Date', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(dateTime, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Analyst', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(analyst, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                  ],
                ),
                pw.SizedBox(height: 18),
                pw.Text('2. Results Summary', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
                pw.SizedBox(height: 6),
                pw.Table(
                  border: pw.TableBorder.all(color: PdfColor.fromHex('#204080'), width: 1.2),
                  columnWidths: {
                    0: const pw.FlexColumnWidth(1),
                    1: const pw.FlexColumnWidth(1),
                  },
                  children: [
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Metric', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Result', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Final HER2 Score', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(report.score, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Confidence (%)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(confidence != null ? confidence.toStringAsFixed(2) : '—', style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Probability (0)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(formatPercent(probabilities['0']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Probability (1+)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(formatPercent(probabilities['1+']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Probability (2+)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(formatPercent(probabilities['2+']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                    pw.TableRow(
                      children: [
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text('Probability (3+)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                        ),
                        pw.Container(
                          alignment: pw.Alignment.centerLeft,
                          padding: const pw.EdgeInsets.all(8),
                          height: 22,
                          child: pw.Text(formatPercent(probabilities['3+']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                        ),
                      ],
                    ),
                  ],
                ),
                pw.SizedBox(height: 18),
                pw.Text('3. Image Analysis', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
                pw.SizedBox(height: 6),
                pw.Text('Below are the visual results produced by the HER2 Classifier system.', style: pw.TextStyle(fontSize: 9)),
                pw.SizedBox(height: 8),
                pw.Row(
                  crossAxisAlignment: pw.CrossAxisAlignment.start,
                  mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
                  children: report.isHneCheckbox == true
                      ? [
                          pw.Column(
                            crossAxisAlignment: pw.CrossAxisAlignment.start,
                            children: [
                              pw.Text('Input Image:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                              pw.SizedBox(height: 4),
                              inputImageBytes != null
                                  ? pw.Image(pw.MemoryImage(inputImageBytes), height: 80, width: 80)
                                  : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                            ],
                          ),
                          pw.Column(
                            crossAxisAlignment: pw.CrossAxisAlignment.center,
                            children: [
                              pw.Text('Synthetic IHC:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                              pw.SizedBox(height: 4),
                              (report.generated_b64 != null && report.generated_b64!.isNotEmpty)
                                  ? pw.Image(pw.MemoryImage(base64Decode(report.generated_b64!)), height: 80, width: 80)
                                  : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                            ],
                          ),
                          pw.Column(
                            crossAxisAlignment: pw.CrossAxisAlignment.start,
                            children: [
                              pw.Text('Pseudo-color Mapping:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                              pw.SizedBox(height: 4),
                              pseudoImages.isNotEmpty
                                  ? pw.Image(pw.MemoryImage(pseudoImages.first), height: 110, width: 110)
                                  : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                            ],
                          ),
                        ]
                      : [
                          pw.Column(
                            crossAxisAlignment: pw.CrossAxisAlignment.start,
                            children: [
                              pw.Text('Input Image:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                              pw.SizedBox(height: 4),
                              inputImageBytes != null
                                  ? pw.Image(pw.MemoryImage(inputImageBytes), height: 80, width: 80)
                                  : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                            ],
                          ),
                          pw.Column(
                            crossAxisAlignment: pw.CrossAxisAlignment.start,
                            children: [
                              pw.Text('Pseudo-color Mapping:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                              pw.SizedBox(height: 4),
                              pseudoImages.isNotEmpty
                                  ? pw.Image(pw.MemoryImage(pseudoImages.first), height: 110, width: 110)
                                  : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                            ],
                          ),
                        ],
                ),
                pw.SizedBox(height: 12),
                pw.Text('Grad-CAM Visualizations:', style: pw.TextStyle(fontSize: 8, fontWeight: pw.FontWeight.bold)),
                pw.SizedBox(height: 6),
                gradCamImages.isNotEmpty
                    ? pw.Wrap(
                        spacing: 6,
                        runSpacing: 6,
                        children: gradCamImages.take(5).map((img) => pw.Image(pw.MemoryImage(img), height: 60, width: 60)).toList(),
                      )
                    : pw.Text('Grad-CAM visualizations are not available for this case.', style: pw.TextStyle(fontSize: 8)),
                pw.SizedBox(height: 18),
                pw.Text('4. Interpretation Notes', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
                pw.SizedBox(height: 6),
                pw.Text(interpretation, style: pw.TextStyle(fontSize: 9)),
                pw.SizedBox(height: 18),
                pw.Divider(color: PdfColor.fromHex('#D9D9D9')),
                pw.Center(
                  child: pw.Text('Generated by HER2 Classifier, Princess Nourah University, For research and educational use only.', style: pw.TextStyle(fontSize: 7)),
                ),
              ],
            );
          },
        ),
      );
    }
    await Printing.layoutPdf(
      onLayout: (format) async => pdf.save(),
    );
  }




Future<List<Report>> loadReports() async {
  final prefs = await SharedPreferences.getInstance();
  final List<String> reports = prefs.getStringList('reports') ?? [];
  return reports.map((e) => Report.fromJson(jsonDecode(e))).toList();
}

class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  Future<void> generatePdf(Report report) async {
    final pdf = pw.Document();
    final ByteData logoData = await rootBundle.load('lib/assets/images/logo.png');
    final Uint8List logoBytes = logoData.buffer.asUint8List();
    String slideName = (report.fileName ?? '-').split('/').last;
    final String imageType = report.isHneCheckbox == true ? 'H&E (converted to IHC)' : 'IHC';
    String imageSize = (report.imageSize ?? '-');
    if (imageSize != '-' && imageSize.contains('x')) {
      imageSize = imageSize.replaceAll(' ', '');
      imageSize = imageSize.replaceAll('×', 'x');
      var parts = imageSize.split('x');
      if (parts.length == 2) {
        imageSize = '${parts[0]} × ${parts[1]} px';
      }
    }
    final String dateTime = report.date.replaceAll('T', ' ');
    final String analyst = 'HER2 Classifier AI Model';
    final List<String> probKeys = ['0', '1+', '2+', '3+'];
    final Map<String, double> probabilities = {};
    if (report.probs != null) {
      for (var k in probKeys) {
        var v = report.probs![k];
        if (v is num) probabilities[k] = v.toDouble();
      }
    }
    double? confidence = report.confidence?.toDouble();
    if (confidence == null && probabilities.isNotEmpty) {
      confidence = probabilities.values.reduce((a, b) => a > b ? a : b);
    }
    String formatPercent(num? v) => v == null ? '—' : '${(v * 100).toStringAsFixed(2)}%';
    Uint8List? inputImageBytes;
    if (report.origB64.isNotEmpty) {
      inputImageBytes = base64Decode(report.origB64);
    }
    Uint8List? generatedImageBytes;
    if (report.generated_b64 != null && report.generated_b64!.isNotEmpty) {
      generatedImageBytes = base64Decode(report.generated_b64!);
    }
    final List<Uint8List> pseudoImages = [];
    if (report.pseudoB64.isNotEmpty) {
      pseudoImages.add(base64Decode(report.pseudoB64));
    }
    final List<Uint8List> gradCamImages = [];
    if (report.gradcamLayers != null && report.gradcamLayers!.isNotEmpty) {
      for (var layer in report.gradcamLayers!) {
        if (layer['gradcam_b64'] != null && (layer['gradcam_b64'] as String).isNotEmpty) {
          gradCamImages.add(base64Decode(layer['gradcam_b64']));
        }
      }
    }
    String interpretation = '';
    switch (report.score) {
      case '0':
        interpretation = 'The AI system detected low HER2 expression corresponding to a score of 0. This indicates minimal or no membrane staining, consistent with a HER2-negative phenotype.';
        break;
      case '1+':
        interpretation = 'The AI system indicates HER2 Score 1+ (negative). Faint/incomplete membrane staining is observed in tumor cells.';
        break;
      case '2+':
        interpretation = 'The AI system indicates HER2 Score 2+ (equivocal). Weak to moderate complete membrane staining may be present. Additional confirmatory assessment is recommended to determine HER2 amplification status.';
        break;
      case '3+':
        interpretation = 'The AI system indicates HER2 Score 3+ (positive). Intense, complete membrane staining is identified in tumor cells.';
        break;
      default:
        interpretation = '—';
    }
    String username = '-';
    if (report.toJson().containsKey('username')) {
      username = report.toJson()['username'] ?? '-';
    } else {
      final prefs = await SharedPreferences.getInstance();
      username = prefs.getString('current_username') ?? '-';
    }
    pdf.addPage(
      pw.Page(
        margin: const pw.EdgeInsets.all(40),
        build: (pw.Context context) {
          return pw.Column(
            crossAxisAlignment: pw.CrossAxisAlignment.start,
            children: [
              pw.Center(
                child: pw.Column(
                  children: [
                    pw.Image(
                      pw.MemoryImage(logoBytes),
                      height: 32,
                    ),
                    pw.SizedBox(height: 8),
                    pw.Text(
                      'HER2 Analysis Report',
                      style: pw.TextStyle(
                        fontSize: 18,
                        fontWeight: pw.FontWeight.bold,
                        color: PdfColor.fromHex('#204080'),
                      ),
                    ),
                  ],
                ),
              ),
              pw.SizedBox(height: 8),
              pw.Center(
                child: pw.Text(
                  'Date: $dateTime | Uploaded By: $username | Analyst: $analyst',
                  style: pw.TextStyle(fontSize: 9),
                ),
              ),
              pw.SizedBox(height: 18),
              // Section 1: Sample Information
              pw.Text('1. Sample Information', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
              pw.SizedBox(height: 6),
              pw.Table(
                border: pw.TableBorder.all(color: PdfColor.fromHex('#204080'), width: 1.2),
                columnWidths: {
                  0: const pw.FlexColumnWidth(1),
                  1: const pw.FlexColumnWidth(1),
                },
                children: [
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('File Name', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(slideName, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Image Type', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(imageType, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Image Size', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(imageSize, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Date', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(dateTime, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Analyst', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(analyst, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                ],
              ),
              pw.SizedBox(height: 18),
              // Section 2: Results Summary
              pw.Text('2. Results Summary', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
              pw.SizedBox(height: 6),
              pw.Table(
                border: pw.TableBorder.all(color: PdfColor.fromHex('#204080'), width: 1.2),
                columnWidths: {
                  0: const pw.FlexColumnWidth(1),
                  1: const pw.FlexColumnWidth(1),
                },
                children: [
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Metric', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Result', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Final HER2 Score', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(report.score, style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Confidence (%)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(confidence != null ? confidence.toStringAsFixed(2) : '—', style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Probability (0)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(formatPercent(probabilities['0']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Probability (1+)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(formatPercent(probabilities['1+']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Probability (2+)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(formatPercent(probabilities['2+']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                  pw.TableRow(
                    children: [
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text('Probability (3+)', style: pw.TextStyle(fontSize: 7, font: pw.Font.helveticaBold())),
                      ),
                      pw.Container(
                        alignment: pw.Alignment.centerLeft,
                        padding: const pw.EdgeInsets.all(8),
                        height: 22,
                        child: pw.Text(formatPercent(probabilities['3+']), style: pw.TextStyle(fontSize: 7, font: pw.Font.helvetica())),
                      ),
                    ],
                  ),
                ],
              ),
              pw.SizedBox(height: 18),
              // Section 3: Image Analysis
              pw.Text('3. Image Analysis', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
              pw.SizedBox(height: 6),
              pw.Text('Below are the visual results produced by the HER2 Classifier system.', style: pw.TextStyle(fontSize: 9)),
              pw.SizedBox(height: 8),
              pw.Row(
                crossAxisAlignment: pw.CrossAxisAlignment.start,
                mainAxisAlignment: pw.MainAxisAlignment.spaceBetween,
                children: report.isHneCheckbox == true
                    ? [
                        pw.Column(
                          crossAxisAlignment: pw.CrossAxisAlignment.start,
                          children: [
                            pw.Text('Input Image:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                            pw.SizedBox(height: 4),
                            inputImageBytes != null
                                ? pw.Image(pw.MemoryImage(inputImageBytes), height: 80, width: 80)
                                : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                          ],
                        ),
                        pw.Column(
                          crossAxisAlignment: pw.CrossAxisAlignment.center,
                          children: [
                            pw.Text('Synthetic IHC:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                            pw.SizedBox(height: 4),
                            (report.generated_b64 != null && report.generated_b64!.isNotEmpty)
                                ? pw.Image(pw.MemoryImage(base64Decode(report.generated_b64!)), height: 80, width: 80)
                                : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                          ],
                        ),
                        pw.Column(
                          crossAxisAlignment: pw.CrossAxisAlignment.start,
                          children: [
                            pw.Text('Pseudo-color Mapping:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                            pw.SizedBox(height: 4),
                            pseudoImages.isNotEmpty
                                ? pw.Image(pw.MemoryImage(pseudoImages.first), height: 110, width: 110)
                                : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                          ],
                        ),
                      ]
                    : [
                        pw.Column(
                          crossAxisAlignment: pw.CrossAxisAlignment.start,
                          children: [
                            pw.Text('Input Image:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                            pw.SizedBox(height: 4),
                            inputImageBytes != null
                                ? pw.Image(pw.MemoryImage(inputImageBytes), height: 80, width: 80)
                                : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                          ],
                        ),
                        pw.Column(
                          crossAxisAlignment: pw.CrossAxisAlignment.start,
                          children: [
                            pw.Text('Pseudo-color Mapping:', style: pw.TextStyle(fontSize: 10, fontWeight: pw.FontWeight.bold)),
                            pw.SizedBox(height: 4),
                            pseudoImages.isNotEmpty
                                ? pw.Image(pw.MemoryImage(pseudoImages.first), height: 110, width: 110)
                                : pw.Container(height: 80, width: 80, decoration: pw.BoxDecoration(border: pw.Border.all(color: PdfColor.fromHex('#204080')))),
                          ],
                        ),
                      ],
              ),
              pw.SizedBox(height: 12),
              pw.Text('Grad-CAM Visualizations:', style: pw.TextStyle(fontSize: 8, fontWeight: pw.FontWeight.bold)),
              pw.SizedBox(height: 6),
              gradCamImages.isNotEmpty
                  ? pw.Wrap(
                      spacing: 6,
                      runSpacing: 6,
                      children: gradCamImages.take(5).map((img) => pw.Image(pw.MemoryImage(img), height: 60, width: 60)).toList(),
                    )
                  : pw.Text('Grad-CAM visualizations are not available for this case.', style: pw.TextStyle(fontSize: 8)),
              pw.SizedBox(height: 18),
              // Section 4: Interpretation Notes
              pw.Text('4. Interpretation Notes', style: pw.TextStyle(fontSize: 9, fontWeight: pw.FontWeight.bold, color: PdfColor.fromHex('#204080'))),
              pw.SizedBox(height: 6),
              pw.Text(interpretation, style: pw.TextStyle(fontSize: 9)),
              pw.SizedBox(height: 18),
              pw.Divider(color: PdfColor.fromHex('#D9D9D9')),
              pw.Center(
                child: pw.Text('Generated by HER2 Classifier, Princess Nourah University, For research and educational use only.', style: pw.TextStyle(fontSize: 6)),
              ),
            ],
          );
        },
      ),
    );
    await Printing.layoutPdf(
      onLayout: (format) async => pdf.save(),
    );
  }
  List<Report> reports = [];
  List<Report> filteredReports = [];
  String _activeRoute = '/history';
  String _searchText = '';
  String? _selectedDate;
  String? _selectedScoreCategory;
  Map<int, bool> selectedReports = {};
  bool selectAll = false;

  @override
  void initState() {
    super.initState();
    loadReports().then((value) {
      setState(() {
        reports = value;
        filteredReports = value;
      });
    });
  }

  void _logout() {
    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
    final bool isDark = themeProvider.themeMode == ThemeMode.dark;
    final Color dialogBg = isDark ? const Color(0xFF1E293B) : Colors.white;
    final Color dialogText = isDark ? const Color(0xFFE5E7EB) : Colors.black;
    final Color accent = const Color(0xFF4F7BFF);

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        backgroundColor: dialogBg,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
        title: Text('Log Out',
            textAlign: TextAlign.center, style: TextStyle(color: dialogText)),
        content: Text('Are you sure you want to logout?',
            textAlign: TextAlign.center, style: TextStyle(color: dialogText)),
        actionsAlignment: MainAxisAlignment.center,
        actions: [
          TextButton(
              onPressed: () => Navigator.pop(context),
              child: Text('Cancel', style: TextStyle(color: accent))),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFE74C3C),
              shape:
                  RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            ),
            onPressed: () {
              
              Navigator.of(context).pushReplacementNamed('/');
            },
            child: const Text('Log Out', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }

  void _navigate(String route) {
    setState(() => _activeRoute = route);
    if (route != '/history') {
      Navigator.of(context).pushReplacementNamed(route);
    }
  }

  void _filterReports() {
    setState(() {
      filteredReports = reports.where((report) {
        final matchesText = _searchText.isEmpty ||
            report.score.toLowerCase().contains(_searchText.toLowerCase()) ||
            _getImageName(report)
                .toLowerCase()
                .contains(_searchText.toLowerCase());
        final matchesDate =
            _selectedDate == null || report.date.startsWith(_selectedDate!);
        final matchesCategory = _selectedScoreCategory == null ||
            report.score == _selectedScoreCategory;
        return matchesText && matchesDate && matchesCategory;
      }).toList();
    });
  }


  String _getImageName(Report report) {
    if (report.fileName != null && report.fileName!.isNotEmpty && report.fileName != '-') {
      return report.fileName!;
    }
    return 'Image';
  }

  List<String> _getAvailableDates() {
    final dates = reports.map((r) => r.date.split('T').first).toSet().toList();
    dates.sort((a, b) => b.compareTo(a));
    return dates;
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final bool isDark = themeProvider.themeMode == ThemeMode.dark;
    final Color accent = const Color(0xFF4F7BFF);
    // Custom background and sidebar colors
  final Color bgColor = isDark ? Theme.of(context).colorScheme.surface : const Color(0xFFF6F8FC);
  final Color sidebarColor = isDark ? const Color(0xFF1E293B) : Colors.white;
  final Color cardColor = isDark ? const Color(0xFF1E293B) : Colors.white;
    final Color textColor = Theme.of(context).colorScheme.onSurface;
    final Color textSecondary = Theme.of(context).colorScheme.onSurface.withOpacity(0.7);
    final Color darkText = const Color(0xFFE5E7EB);
    final Color lightText = Colors.black;
    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Sidebar(
              onLogout: _logout,
              onNavigate: _navigate,
              activeRoute: _activeRoute,
              backgroundColor: sidebarColor,
              accentColor: accent,
              textColor: textColor,
              secondaryTextColor: textSecondary,
            ),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.all(24.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const SizedBox(height: 16),
                    Text(
                      "Analysis History",
            style: TextStyle(
              fontSize: 28,
              fontWeight: FontWeight.bold,
              color: textColor),
                    ),
                    const SizedBox(height: 24),

                    Row(
                      crossAxisAlignment: CrossAxisAlignment.center,
                      children: [
                        Expanded(
                          child: TextField(
                            onChanged: (val) {
                              _searchText = val;
                              _filterReports();
                            },
                            decoration: InputDecoration(
                              hintText: 'Search by score or image name...',
                              filled: true,
                              fillColor: cardColor,
                              prefixIcon: Icon(Icons.search, color: accent),
                              border: OutlineInputBorder(
                                borderRadius: BorderRadius.circular(14),
                                borderSide:
                                    BorderSide(color: accent.withOpacity(0.2)),
                              ),
                              contentPadding: const EdgeInsets.symmetric(
                                  vertical: 0, horizontal: 16),
                            ),
              style:
                TextStyle(color: textColor),
                          ),
                        ),
                        const SizedBox(width: 10),
                        DropdownButton<String?>(
                          value: _selectedDate,
                          hint: const Text('All Dates'),
                          items: <DropdownMenuItem<String?>>[
                            const DropdownMenuItem<String?>(
                                value: null, child: Text('All Dates')),
                            ..._getAvailableDates().map(
                              (date) => DropdownMenuItem<String?>(
                                  value: date, child: Text(date)),
                            ),
                          ],
                          onChanged: (val) {
                            setState(() {
                              _selectedDate = val;
                              _filterReports();
                            });
                          },
              style:
                TextStyle(color: textColor),
              dropdownColor: cardColor,
                        ),
                        const SizedBox(width: 10),
                        DropdownButton<String?>(
                          value: _selectedScoreCategory,
                          hint: const Text('All Categories'),
                          items: const <DropdownMenuItem<String?>>[
                            DropdownMenuItem<String?>(
                                value: null, child: Text('All Categories')),
                            DropdownMenuItem<String?>(
                                value: '3+', child: Text('3+')),
                            DropdownMenuItem<String?>(
                                value: '2+', child: Text('2+')),
                            DropdownMenuItem<String?>(
                                value: '1+', child: Text('1+')),
                            DropdownMenuItem<String?>(
                                value: '0', child: Text('0')),
                          ],
                          onChanged: (val) {
                            setState(() {
                              _selectedScoreCategory = val;
                              _filterReports();
                            });
                          },
              style:
                TextStyle(color: textColor),
              dropdownColor: cardColor,
                        ),
                        const SizedBox(width: 10),
                        PopupMenuButton<String>(
                          icon: Icon(Icons.more_vert, color: accent),
                          color: cardColor,
                          itemBuilder: (context) => [
                            PopupMenuItem(
                              value: 'select_all',
                              child: Row(
                                children: [
                                  Icon(Icons.select_all, color: accent),
                                  const SizedBox(width: 8),
                                  const Text('Select All'),
                                ],
                              ),
                            ),
                            PopupMenuItem(
                              value: 'delete',
                              enabled: selectedReports.containsValue(true),
                              child: Row(
                                children: [
                                  Icon(Icons.delete, color: accent),
                                  const SizedBox(width: 8),
                                  const Text('Delete'),
                                ],
                              ),
                            ),
                            PopupMenuItem(
                              value: 'print',
                              enabled: selectedReports.containsValue(true),
                              child: Row(
                                children: [
                                  Icon(Icons.print, color: accent),
                                  const SizedBox(width: 8),
                                  const Text('Print'),
                                ],
                              ),
                            ),
                          ],
                          onSelected: (value) {
                            if (value == 'select_all') {
                              setState(() {
                                selectAll = true;
                                for (int i = 0; i < filteredReports.length; i++) {
                                  selectedReports[i] = true;
                                }
                              });
                            } else if (value == 'delete') {
                              setState(() {
                                final toDeleteReports = selectedReports.entries
                                    .where((e) => e.value)
                                    .map((e) => filteredReports[e.key])
                                    .toList();
                                reports.removeWhere((r) => toDeleteReports.contains(r));
                                filteredReports.removeWhere((r) => toDeleteReports.contains(r));
                                selectedReports.clear();
                                selectAll = false;
                              });
                              SharedPreferences.getInstance().then((prefs) {
                                final List<String> reportsJson = reports.map((r) => jsonEncode(r.toJson())).toList();
                                prefs.setStringList('reports', reportsJson);
                              });
                            } else if (value == 'print') {
                              final selected = selectedReports.entries.where((e) => e.value).map((e) => filteredReports[e.key]).toList();
                              if (selected.isNotEmpty) {
                                if (selected.length == 1) {
                                  generatePdf(selected.first);
                                } else {
                                  printSelectedReports(selected);
                                }
                              }
                            }
                          },
                        ),
                      ],
                    ),

                    const SizedBox(height: 24),

                    Expanded(
                      child: filteredReports.isEmpty
                          ? Center(
                              child: Text(
                                "No reports found",
                                style: TextStyle(
                                    color: isDark ? darkText : lightText,
                                    fontSize: 16),
                              ),
                            )
                          : ListView.builder(
                              itemCount: filteredReports.length,
                              itemBuilder: (context, index) {
                                final report = filteredReports[index];
                                final dateStr = report.date.split('T').first;
                                final score = report.score;

                                Color scoreColor =
                                    isDark ? darkText : lightText;
                                if (score.contains('3+')) {
                                  scoreColor = Colors.red;
                                } else if (score.contains('2+')) {
                                  scoreColor = Colors.orange;
                                } else if (score.contains('1+')) {
                                  scoreColor = Colors.amber[800]!;
                                } else if (score == '0') {
                                  scoreColor = Colors.black;
                                }

                                return Card(
                                  margin: const EdgeInsets.symmetric(
                                      vertical: 10, horizontal: 0),
                                  color: cardColor,
                                  elevation: isDark ? 2 : 4,
                                  shadowColor: isDark
                                      ? Colors.black12
                                      : Colors.grey.withOpacity(0.10),
                                  shape: RoundedRectangleBorder(
                                      borderRadius: BorderRadius.circular(16)),
                                  child: Padding(
                                    padding: const EdgeInsets.symmetric(
                                        vertical: 14, horizontal: 24),
                                    child: Row(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.center,
                                      children: [
                                        Checkbox(
                                          value: selectedReports[index] ?? false,
                                          onChanged: (val) {
                                            setState(() {
                                              selectedReports[index] =
                                                  val ?? false;
                                              selectAll = selectedReports
                                                      .length ==
                                                  filteredReports.length &&
                                                  selectedReports.values
                                                      .every((v) => v);
                                            });
                                          },
                      checkColor: Colors.white,
                      activeColor: accent,
                                        ),
                                        if (report.origB64.isNotEmpty)
                                          Padding(
                                            padding: const EdgeInsets.only(
                                                right: 14.0),
                                            child: ClipRRect(
                                              borderRadius:
                                                  BorderRadius.circular(10),
                                              child: Image.memory(
                                                base64Decode(report.origB64),
                                                width: 44,
                                                height: 44,
                                                fit: BoxFit.cover,
                                              ),
                                            ),
                                          ),
                                        Expanded(
                                          child: Column(
                                            crossAxisAlignment:
                                                CrossAxisAlignment.start,
                                            children: [
                                              Text(
                                                dateStr,
                                                style: TextStyle(
                                                  fontSize: 15,
                                                  color: isDark
                                                      ? textSecondary
                                                      : textSecondary,
                                                ),
                                              ),
                                              const SizedBox(height: 2),
                                              Row(
                                                children: [
                                                  Text(
                                                    'HER2 Score: ',
                                                    style: TextStyle(
                                                      fontWeight:
                                                          FontWeight.bold,
                                                      color: isDark
                                                          ? darkText
                                                          : lightText,
                                                      fontSize: 16,
                                                    ),
                                                  ),
                                                  Text(
                                                    score,
                                                    style: TextStyle(
                                                      fontWeight:
                                                          FontWeight.bold,
                                                      color: scoreColor,
                                                      fontSize: 16,
                                                    ),
                                                  ),
                                                ],
                                              ),
                                              const SizedBox(height: 2),
                                              Text(
                                                _getImageName(report),
                                                style: TextStyle(
                                                  fontSize: 13,
                                                  color: isDark
                                                      ? textSecondary
                                                      : textSecondary,
                                                ),
                                              ),
                                            ],
                                          ),
                                        ),
                                        const SizedBox(width: 18),
                                        ElevatedButton(
                                          style: ElevatedButton.styleFrom(
                                            backgroundColor: accent,
                                            shape: RoundedRectangleBorder(
                                              borderRadius:
                                                  BorderRadius.circular(14),
                                            ),
                                            padding: const EdgeInsets.symmetric(
                                                horizontal: 28, vertical: 12),
                                            elevation: 0,
                                          ).copyWith(
                                            overlayColor:
                                                WidgetStateProperty.resolveWith<
                                                    Color?>(
                                              (states) {
                                                if (states.contains(
                                                    WidgetState.hovered)) {
                                                  return const Color(
                                                      0xFF7DC9FF);
                                                }
                                                if (states.contains(
                                                    WidgetState.pressed)) {
                                                  return const Color(
                                                      0xFF3B5EDC);
                                                }
                                                return null;
                                              },
                                            ),
                                          ),
                                          onPressed: () {
                                            List<Map<String, dynamic>>? gradcamLayers;
                                            if (report.gradcamLayers != null) {
                                              if (report.gradcamLayers is List<Map<String, dynamic>>) {
                                                gradcamLayers = report.gradcamLayers;
                                              } else if (report.gradcamLayers is List) {
                                                gradcamLayers = List<Map<String, dynamic>>.from(report.gradcamLayers!.map((e) => Map<String, dynamic>.from(e)));
                                              }
                                            }
                                            String? primaryGradcamB64 = report.primaryGradcamB64;
                                            if ((primaryGradcamB64 == null || primaryGradcamB64.isEmpty) && report.toJson().containsKey('primary_gradcam_b64')) {
                                              primaryGradcamB64 = report.toJson()['primary_gradcam_b64'] as String?;
                                            }
                                             // Force isHneCheckbox to true if generated_b64 exists
                                             bool? isHneCheckbox = report.isHneCheckbox;
                                             if ((report.generated_b64 != null && report.generated_b64!.isNotEmpty)) {
                                               isHneCheckbox = true;
                                             }
                                            Navigator.push(
                                              context,
                                              MaterialPageRoute(
                                                builder: (context) => ResultScreen(
                                                  her2Score: report.score,
                                                  gradcamLayers: gradcamLayers,
                                                  primaryGradcamB64: primaryGradcamB64,
                                                  pseudoB64: report.pseudoB64,
                                                  probs: report.probs ?? {},
                                                  origB64: report.origB64,
                                                  confidence: report.confidence ?? 0.0,
                                                  fileName: report.fileName ?? '-',
                                                  imageSize: report.imageSize ?? '-',
                                                  analysisDate: report.date,
                                                  probs_chart_b64: report.probs_chart_b64,
                                                  generated_b64: report.generated_b64,
                                                    isHneCheckbox: isHneCheckbox,
                                                ),
                                                settings: RouteSettings(arguments: 'fromHistory'),
                                              ),
                                            );
                                          },
                                          child: const Text(
                                            "View Report",
                                            style: TextStyle(
                                              color: Colors.white,
                                              fontWeight: FontWeight.bold,
                                              fontSize: 15,
                                            ),
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                );
                              },
                            ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

}

