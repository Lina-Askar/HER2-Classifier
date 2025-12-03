library;
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../widgets/sidebar.dart';
import 'Report.dart';
import '../theme_provider.dart';
class PopoverInfoIcon extends StatefulWidget {
  final String title;
  final String desc;
  final String caption;
  final bool isDark;
  const PopoverInfoIcon({
    super.key,
    required this.title,
    required this.desc,
    required this.caption,
    required this.isDark,
  });

  @override
  State<PopoverInfoIcon> createState() => _PopoverInfoIconState();
}

class _PopoverInfoIconState extends State<PopoverInfoIcon> {
  OverlayEntry? _popoverEntry;

  void _showPopover() {
    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset position = box.localToGlobal(Offset.zero);
    _popoverEntry = OverlayEntry(
      builder: (context) => Positioned(
        left: position.dx - 100,
        top: position.dy + box.size.height + 8,
        child: Material(
          color: Colors.transparent,
          child: Container(
            width: 240,
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: widget.isDark ? const Color(0xFF1E293B) : Colors.white,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.12),
                  blurRadius: 16,
                  offset: const Offset(0, 4),
                ),
              ],
              border: Border.all(color: const Color(0xFF4F7BFF), width: 1.5),
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                Text(widget.title,
                    style: const TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                        color: Color(0xFF4F7BFF))),
                const SizedBox(height: 8),
                Text(widget.desc,
                    style: TextStyle(
                        fontSize: 10,
                        color: widget.isDark
                            ? const Color(0xFFE5E7EB)
                            : const Color(0xFF4F7BFF),
                        height: 1.5),
                    textAlign: TextAlign.center),
                const SizedBox(height: 6),
                Text(widget.caption,
                    style: TextStyle(
                        fontSize: 9,
                        color: widget.isDark
                            ? const Color(0xFFE5E7EB)
                            : const Color(0xFF4F7BFF),
                        height: 1.4),
                    textAlign: TextAlign.center),
                const SizedBox(height: 8),
                GestureDetector(
                  onTap: _hidePopover,
                  child: const Text("Close",
                      style: TextStyle(
                          color: Color(0xFF4F7BFF),
                          fontWeight: FontWeight.w600,
                          fontSize: 13)),
                )
              ],
            ),
          ),
        ),
      ),
    );
    Overlay.of(context).insert(_popoverEntry!);
  }

  void _hidePopover() {
    _popoverEntry?.remove();
    _popoverEntry = null;
  }

  @override
  void dispose() {
    _hidePopover();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () {
        if (_popoverEntry == null) {
          _showPopover();
        } else {
          _hidePopover();
        }
      },
      child:
          const Icon(Icons.info_outline, color: Color(0xFF4F7BFF), size: 18),
    );
  }
}

class ResultScreen extends StatefulWidget {
      final String? origResizedB64;
    final String? wsi_overlay_b64;
    final Map<String, dynamic>? wsi_box;

  final String her2Score;
  final List<Map<String, dynamic>>? gradcamLayers;
  final String? primaryGradcamB64;
  final String? pseudoB64;
  final Map<String, dynamic>? probs;
  final String? origB64;
  final num? confidence;
  final String fileName;
  final String imageSize;
  final String analysisDate;
  final String? probs_chart_b64;
  final bool? isHneCheckbox;
  final String? generated_b64; 

  const ResultScreen({
    super.key,
    this.probs_chart_b64,
    this.gradcamLayers,
    this.primaryGradcamB64,
    this.pseudoB64,
    this.probs,
    this.origB64,
    this.generated_b64, 
    this.confidence,
    this.fileName = '-',
    this.imageSize = '-',
    this.analysisDate = '-',
    required this.her2Score,
    this.isHneCheckbox,
    this.wsi_overlay_b64,
    this.wsi_box,
    this.origResizedB64,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
    /// ====== Overlay with Box (Full Image) ======
    Widget overlayWithBox({String? overlayB64, Map<String, dynamic>? box, double height = 160, double width = 235, Color boxColor = Colors.red}) {
      if (overlayB64 == null || overlayB64.isEmpty) {
        return safeImage(widget.origB64, height: height);
      }
      try {
        final image = Image.memory(base64Decode(overlayB64), fit: BoxFit.contain, height: height, width: width);
        if (box == null || box.isEmpty) {
          return image;
        }
        return Stack(
          children: [
            image,
            Positioned(
              left: (box['x0'] ?? 0) * width / (box['img_w'] ?? width),
              top: (box['y0'] ?? 0) * height / (box['img_h'] ?? height),
              child: Container(
                width: ((box['x1'] ?? 0) - (box['x0'] ?? 0)) * width / (box['img_w'] ?? width),
                height: ((box['y1'] ?? 0) - (box['y0'] ?? 0)) * height / (box['img_h'] ?? height),
                decoration: BoxDecoration(
                  border: Border.all(color: boxColor, width: 3),
                ),
              ),
            ),
          ],
        );
      } catch (_) {
        return safeImage(widget.origB64, height: height);
      }
    }

    void showImageDialog({required String? b64, Map<String, dynamic>? box, Color boxColor = Colors.red, String? title}) {
      if (b64 == null || b64.isEmpty) return;
      showDialog(
        context: context,
        builder: (context) {
          return Dialog(
            backgroundColor: Colors.transparent,
            child: Container(
              padding: const EdgeInsets.all(12),
              color: Colors.white,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (title != null) ...[
                    Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
                    const SizedBox(height: 8),
                  ],
                  overlayWithBox(
                    overlayB64: b64,
                    box: box,
                    height: 400,
                    width: 600,
                    boxColor: boxColor,
                  ),
                  const SizedBox(height: 8),
                  TextButton(
                    onPressed: () => Navigator.pop(context),
                    child: const Text('Close'),
                  ),
                ],
              ),
            ),
          );
        },
      );
    }
  @override
  void initState() {
    super.initState();
    // Save report only if not from History page
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final ModalRoute? route = ModalRoute.of(context);
      if (route != null && route.settings.arguments != 'fromHistory') {
        _saveReportToHistory();
      }
    });
  }

  Future<void> _saveReportToHistory() async {
    debugPrint('Saving primaryGradcamB64: (length = [38;5;28m${widget.primaryGradcamB64?.length}[0m)');
    final prefs = await SharedPreferences.getInstance();
    final List<String> reports = prefs.getStringList('reports') ?? [];

    // Check for reports with same file name
    String baseFileName = widget.fileName;
    String newFileName = baseFileName;
    int count = 1;
    final existingNames = reports.map((e) {
      try {
        final r = Report.fromJson(jsonDecode(e));
        return r.fileName;
      } catch (_) {
        return null;
      }
    }).where((name) => name != null).toList();

    while (existingNames.contains(newFileName)) {
      final dotIndex = baseFileName.lastIndexOf('.');
      if (dotIndex != -1) {
        newFileName = '${baseFileName.substring(0, dotIndex)} (${count + 1})${baseFileName.substring(dotIndex)}';
      } else {
        newFileName = '$baseFileName (${count + 1})';
      }
      count++;
    }

    final report = Report(
      date: widget.analysisDate,
      score: widget.her2Score,
      origB64: widget.origB64 ?? '',
      pseudoB64: widget.pseudoB64 ?? '',
      gradcamLayers: widget.gradcamLayers,
      primaryGradcamB64: widget.primaryGradcamB64,
      probs: widget.probs,
      confidence: widget.confidence,
      fileName: newFileName,
      imageSize: widget.imageSize,
      probs_chart_b64: widget.probs_chart_b64,
      isHneCheckbox: widget.isHneCheckbox,
      generated_b64: widget.generated_b64 ?? '', // Ensure generated_b64 is saved
    );
    final reportJson = jsonEncode(report.toJson());
    final alreadyExists = reports.any((e) {
      try {
        final r = Report.fromJson(jsonDecode(e));
        return r.date == report.date && r.fileName == report.fileName;
      } catch (_) {
        return false;
      }
    });
    if (!alreadyExists) {
      reports.add(reportJson);
      await prefs.setStringList('reports', reports);
    }
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
        title: Text('Log Out', textAlign: TextAlign.center, style: TextStyle(color: dialogText)),
        content: Text('Are you sure you want to logout?', textAlign: TextAlign.center, style: TextStyle(color: dialogText)),
        actionsAlignment: MainAxisAlignment.center,
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('Cancel', style: TextStyle(color: accent)),
          ),
          ElevatedButton(
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFFE74C3C),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
            ),
            onPressed: () {
              Navigator.pop(context);
              Navigator.of(context).pushReplacementNamed('/');
            },
            child: const Text('Log Out', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }
  final GlobalKey _scoreBoxKey = GlobalKey();
  double _scoreBoxHeight = 420;
  int _selectedGradCamIndex = 0;
  int _selectedInputIndex = 0;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final ctx = _scoreBoxKey.currentContext;
      if (ctx != null) {
        final box = ctx.findRenderObject() as RenderBox?;
        if (box != null) {
          setState(() {
            _scoreBoxHeight = box.size.height;
          });
        }
      }
    });
  }

  Color _getScoreColor(String score) {
    if (score.contains('3+')) return Colors.red;
    if (score.contains('2+')) return Colors.orange;
    if (score.contains('1+')) return Colors.amber[800]!;
    return Colors.black;
  }

  /// ====== Safe Image ======
  Widget safeImage(String? b64, {double height = 160}) {
    if (b64 == null || b64.isEmpty) {
      return Container(
        height: height,
        width: 235,
        color: Colors.grey[200],
        child: const Center(
            child: Text('No Image', style: TextStyle(color: Colors.black45))),
      );
    }
    try {
      return Image.memory(base64Decode(b64),
          fit: BoxFit.contain, height: height, width: 235);
    } catch (_) {
      return Container(
        height: height,
        width: 235,
        color: Colors.grey[200],
        child: const Center(
            child: Text('Invalid Image', style: TextStyle(color: Colors.red))),
      );
    }
  }

  /// ====== Grad-CAM Card ======
  Widget _gradCamCard(List<Map<String, dynamic>> layers) {
    const accent = Color(0xFF4F7BFF);
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final darkText = isDark ? Color(0xFFE5E7EB) : Color(0xFF111827);
    final cardColor = isDark ? Color(0xFF1E293B) : Colors.white;

    final selectedIndex =
        (_selectedGradCamIndex < layers.length) ? _selectedGradCamIndex : 0;

    return Container(
      width: 275,
      height: 260,
      padding: const EdgeInsets.all(20),
      margin: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
              color: isDark ? Colors.black.withOpacity(0.18) : Colors.black.withOpacity(0.06),
              blurRadius: 12,
              offset: const Offset(0, 2)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text('Grad-CAM',
                  style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.bold,
                      color: darkText)),
              const SizedBox(width: 4),
              PopoverInfoIcon(
                title: 'Grad-CAM',
                desc: 'Visualizes what each layer detects in the image.',
                caption : '',
                isDark: isDark,
              ),
            ],
          ),
          const SizedBox(height: 12),
          if (layers.isEmpty)
            Container(
              height: 160,
              color: Colors.grey[200],
              child: const Center(child: Text('No Grad-CAM Available')),
            )
          else
            _imageCard(
              'Real Image',
              widget.origResizedB64 != null && widget.origResizedB64!.isNotEmpty
                  ? Image.memory(
                      base64Decode(widget.origResizedB64!),
                      fit: BoxFit.contain,
                      height: 160,
                      width: 235,
                    )
                  : (widget.origB64 != null && widget.origB64!.isNotEmpty
                      ? Image.memory(
                          base64Decode(widget.origB64!),
                          fit: BoxFit.contain,
                          height: 160,
                          width: 235,
                        )
                      : (widget.wsi_overlay_b64 != null && widget.wsi_overlay_b64!.isNotEmpty
                          ? Image.memory(
                              base64Decode(widget.wsi_overlay_b64!),
                              fit: BoxFit.contain,
                              height: 160,
                              width: 235,
                            )
                          : Container(
                              height: 160,
                              width: 235,
                              color: Colors.grey[200],
                              child: const Center(child: Text('No Image', style: TextStyle(color: Colors.black45)) ),
                            )
                        )
                    ),
              info: 'Original IHC image'
            ),
          ...List.generate(layers.length, (i) {
            final img = layers[i]['gradcam_b64'];
            final isFirst = i == 0;
            return GestureDetector(
              onTap: () {
                setState(() {
                  _selectedGradCamIndex = i;
                });
              },
              child: Container(
                margin: const EdgeInsets.symmetric(vertical: 3),
                decoration: BoxDecoration(
                  border: Border.all(
                      color: i == selectedIndex
                          ? accent
                          : Colors.white,
                      width: 2),
                  borderRadius: BorderRadius.circular(6),
                ),
                child: Image.memory(
                  base64Decode(img),
                  fit: BoxFit.cover,
                  height: 20,
                  width: 20,
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  /// ====== Image Card ======
  Widget _imageCard(String title, Widget child, {String? info}) {
    const accent = Color(0xFF4F7BFF);
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final darkText = isDark ? Color(0xFFE5E7EB) : Color(0xFF111827);
    final cardColor = isDark ? Color(0xFF1E293B) : Colors.white;

    String desc = '';
    String caption = '';
    if (title == 'Input Image') {
      desc = 'Original IHC image showing HER2 protein.';
      caption = '';
      if (widget.fileName != '-' && widget.fileName.isNotEmpty) {
        caption += 'File: ${widget.fileName}\n';
      }
      if (widget.imageSize != '-' && widget.imageSize.isNotEmpty) {
        caption += 'Size: ${widget.imageSize}\n';
      }
      if (widget.analysisDate != '-' && widget.analysisDate.isNotEmpty) {
        caption += 'Date: ${widget.analysisDate}';
      }
      if (caption.isEmpty) caption = 'The actual uploaded sample.';
    } else if (title == 'Pseudo-color Mapping') {
      desc = 'Visualizes HER2 intensity using heatmap colors.';
     
    }

    return Container(
      width: 275,
      height: 260,
      padding: const EdgeInsets.all(20),
      margin: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
              color: isDark ? Colors.black.withOpacity(0.18) : Colors.black.withOpacity(0.06),
              blurRadius: 12,
              offset: const Offset(0, 2)),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(title,
                  style: TextStyle(
                      fontSize: 17,
                      fontWeight: FontWeight.bold,
                      color: darkText)),
              const SizedBox(width: 4),
              if (info != null)
               PopoverInfoIcon(
                  title: title,
                  desc: desc,
                  caption: caption,
                  isDark: isDark,
                ),
            ],
          ),
          const SizedBox(height: 10),
          Expanded(child: Center(child: child)),
        ],
      ),
    );
  }

  Widget inputImageCard(String? origB64, String? generated_b64) {
    const accent = Color(0xFF4F7BFF);
    const darkText = Color(0xFF111827);
    const cardColor = Colors.white;

    final List<String?> imgs = [origB64, generated_b64];
    final List<String> inputTitles = ["Input Image", "Synthetic IHC"];
    final List<String> descs = [
      'Original H&E image showing the tissue structure before IHC conversion.',
      'Synthetic IHC image generated from H&E.'
    ];
    final List<String> captions = [
      (widget.fileName != '-' && widget.fileName.isNotEmpty ? 'File: ${widget.fileName}\n' : '') +
      (widget.imageSize != '-' && widget.imageSize.isNotEmpty ? 'Size: ${widget.imageSize}\n' : '') +
      (widget.analysisDate != '-' && widget.analysisDate.isNotEmpty ? 'Date: ${widget.analysisDate}' : '') +
      (((widget.fileName == '-' || widget.fileName.isEmpty) && (widget.imageSize == '-' || widget.imageSize.isEmpty) && (widget.analysisDate == '-' || widget.analysisDate.isEmpty)) ? 'The actual uploaded sample.' : ''),
      'This image was generated using PSPStain.'
    ];
    final selected = _selectedInputIndex;

    return Container(
      width: 260,
      height: 260,
      padding: const EdgeInsets.all(20),
      margin: const EdgeInsets.symmetric(horizontal: 12),
      decoration: BoxDecoration(
        color: cardColor,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: Colors.black12,
            blurRadius: 12,
            offset: Offset(0, 2),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                inputTitles[selected],
                style: const TextStyle(
                  fontSize: 17,
                  fontWeight: FontWeight.bold,
                  color: darkText,
                ),
              ),
              const SizedBox(width: 4),
              PopoverInfoIcon(
                title: inputTitles[selected],
                desc: descs[selected],
                caption: captions[selected],
                isDark: false,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Expanded(
            child: Row(
              children: [
                Expanded(
                  flex: 3,
                  child: GestureDetector(
                    onTap: () {
                      if (imgs[selected] != null && imgs[selected]!.isNotEmpty) {
                        showDialog(
                          context: context,
                          builder: (context) => Dialog(
                            backgroundColor: Colors.transparent,
                            child: InteractiveViewer(
                              child: Image.memory(
                                base64Decode(imgs[selected]!),
                                fit: BoxFit.contain,
                              ),
                            ),
                          ),
                        );
                      }
                    },
                    child: Container(
                      margin: const EdgeInsets.only(right: 12),
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.white, width: 2),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: (imgs[selected] == null || imgs[selected]!.isEmpty)
                          ? Container(
                              color: Colors.grey[200],
                              child: const Center(child: Text("No Image")),
                            )
                          : Image.memory(
                              base64Decode(imgs[selected]!),
                              fit: BoxFit.contain,
                              height: 160,
                              width: 235,
                            ),
                    ),
                  ),
                ),
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(imgs.length, (i) {
                    final img = imgs[i];
                    return GestureDetector(
                      onTap: () {
                        setState(() => _selectedInputIndex = i);
                      },
                      child: Container(
                        margin: const EdgeInsets.symmetric(vertical: 3),
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: i == selected ? accent : Colors.white,
                            width: 2,
                          ),
                          borderRadius: BorderRadius.circular(6),
                        ),
                        child: (img == null || img.isEmpty)
                            ? Container(
                                height: 20,
                                width: 20,
                                color: Colors.grey[200],
                                child: Center(
                                  child: Text(
                                    inputTitles[i],
                                    style: const TextStyle(
                                        fontSize: 6, color: Colors.black54),
                                  ),
                                ),
                              )
                            : Image.memory(
                                base64Decode(img),
                                fit: BoxFit.cover,
                                height: 22,
                                width: 22,
                              ),
                      ),
                    );
                  }),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final isDark = themeProvider.themeMode == ThemeMode.dark;
    final accent = Color(0xFF4F7BFF);
    final darkText = isDark ? Color(0xFFE5E7EB) : Color(0xFF111827);
    final grayText = isDark ? Color(0xFFCBD5E1) : Color(0xFF374151);
    final cardColor = isDark ? Color(0xFF1E293B) : Colors.white;
    final bgColor = isDark ? Color(0xFF111827) : Color(0xFFF9FAFB);
    final probsChartB64 = widget.probs_chart_b64;

    return Scaffold(
      backgroundColor: bgColor,
      body: SafeArea(
        child: Row(
          children: [
            Sidebar(
              onLogout: _logout,
              onNavigate: (r) => Navigator.pushNamed(context, r),
              activeRoute: '/result',
              backgroundColor: cardColor,
              accentColor: accent,
              textColor: darkText,
              secondaryTextColor: grayText,
            ),
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    /// ===== Top row =====
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          key: _scoreBoxKey,
                          width: 520,
                          padding: const EdgeInsets.all(28),
                          margin: const EdgeInsets.only(right: 32),
                          decoration: BoxDecoration(
                            color: cardColor,
                            borderRadius: BorderRadius.circular(18),
                            boxShadow: [
                              BoxShadow(
                                  color: Colors.black.withOpacity(0.08),
                                  blurRadius: 12,
                                  offset: const Offset(0, 2)),
                            ],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                Row(children: [
                Text('Score : ',
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: darkText)),
                Text(widget.her2Score,
                  style: TextStyle(
                    fontSize: 34,
                    fontWeight: FontWeight.bold,
                    color: _getScoreColor(widget.her2Score))),
                ]),
                              const SizedBox(height: 18),
                              probsChartB64 != null
                                  ? Image.memory(
                                      base64Decode(probsChartB64),
                                      fit: BoxFit.contain,
                                      width: 420,
                                      height: 180,
                                    )
                                  : const SizedBox(height: 180),
                            ],
                          ),
                        ),
                        Container(
                          width: 320,
                          height: _scoreBoxHeight,
                          padding: const EdgeInsets.all(28),
                          decoration: BoxDecoration(
                            color: cardColor,
                            borderRadius: BorderRadius.circular(18),
                            boxShadow: [
                              BoxShadow(
                                  color: Colors.black.withOpacity(0.08),
                                  blurRadius: 12,
                                  offset: const Offset(0, 2)),
                            ],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                           Text('Confidence : ',
                                  style: TextStyle(
                                      fontSize: 32,
                                      fontWeight: FontWeight.bold,
                      color: darkText)),
                              const SizedBox(height: 18),
                              Center(
                                child: GaugeWidget(
                                  value: (widget.confidence ?? 0.0).toDouble(),
                                  accent: accent,
                                  darkText: darkText,
                                ),
                              ),
                              const SizedBox(height: 8),
                              Center(
                                child: Text(
                                  'Indicates how confident the model is\nin its result.',
                                  textAlign: TextAlign.center,
                                  style: TextStyle(fontSize: 12, color: grayText),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),

                    const SizedBox(height: 40),

                    /// ===== Bottom row =====
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // ===== Input Image Card =====
                        (widget.isHneCheckbox == true)
                          ? inputImageCard(widget.origB64, widget.generated_b64)
                          : _imageCard(
                              'Input Image',
                              widget.origResizedB64 != null && widget.origResizedB64!.isNotEmpty
                                  ? Image.memory(
                                      base64Decode(widget.origResizedB64!),
                                      fit: BoxFit.contain,
                                      height: 160,
                                      width: 235,
                                    )
                                  : (widget.origB64 != null && widget.origB64!.isNotEmpty
                                      ? Image.memory(
                                          base64Decode(widget.origB64!),
                                          fit: BoxFit.contain,
                                          height: 160,
                                          width: 235,
                                        )
                                      : (widget.wsi_overlay_b64 != null && widget.wsi_overlay_b64!.isNotEmpty
                                          ? Image.memory(
                                              base64Decode(widget.wsi_overlay_b64!),
                                              fit: BoxFit.contain,
                                              height: 160,
                                              width: 235,
                                            )
                                          : Container(
                                              height: 160,
                                              width: 235,
                                              color: Colors.grey[200],
                                              child: const Center(child: Text('No Image', style: TextStyle(color: Colors.black45)) ),
                                            )
                                        )
                                  ),
                              info: 'Original image',
                            ),
                        // ===== Grad-CAM Card =====
                        Container(
                          width: 275,
                          height: 260,
                          padding: const EdgeInsets.all(20),
                          margin: const EdgeInsets.symmetric(horizontal: 12),
                          decoration: BoxDecoration(
                            color: cardColor,
                            borderRadius: BorderRadius.circular(18),
                            boxShadow: [
                              BoxShadow(
                                  color: isDark ? Colors.black.withOpacity(0.18) : Colors.black.withOpacity(0.06),
                                  blurRadius: 12,
                                  offset: const Offset(0, 2)),
                            ],
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Text('Grad-CAM',
                                      style: TextStyle(
                                          fontSize: 17,
                                          fontWeight: FontWeight.bold,
                                          color: darkText)),
                                  const SizedBox(width: 4),
                                  PopoverInfoIcon(
                                    title: 'Grad-CAM',
                                    desc: 'Highlights where the model is focusing.',
                                    caption: '',
                                    isDark: isDark,
                                  ),
                                ],
                              ),
                              const SizedBox(height: 12),
                              Expanded(
                                child: Center(
                                  child: GestureDetector(
                                    onTap: () {
                                      if (widget.primaryGradcamB64 != null && widget.primaryGradcamB64!.isNotEmpty) {
                                        showDialog(
                                          context: context,
                                          builder: (context) => Dialog(
                                            backgroundColor: Colors.transparent,
                                            child: InteractiveViewer(
                                              child: Image.memory(
                                                base64Decode(widget.primaryGradcamB64!),
                                                fit: BoxFit.contain,
                                              ),
                                            ),
                                          ),
                                        );
                                      }
                                    },
                                    child: (
                                      widget.primaryGradcamB64 != null && widget.primaryGradcamB64!.isNotEmpty
                                      ? Image.memory(
                                          base64Decode(widget.primaryGradcamB64!),
                                          fit: BoxFit.contain,
                                          height: 160,
                                          width: 235,
                                        )
                                      : Container(
                                          height: 160,
                                          color: Colors.grey[200],
                                          child: const Center(
                                            child: Text(
                                              '  unavilable Grad-CAM    .',
                                              style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold),
                                            ),
                                          ),
                                        )
                                    ),
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                        // ===== Pseudo-color Mapping Card =====
                        _imageCard(
                          'Pseudo-color Mapping',
                          GestureDetector(
                            onTap: () {
                              if (widget.pseudoB64 != null && widget.pseudoB64!.isNotEmpty) {
                                showDialog(
                                  context: context,
                                  builder: (context) => Dialog(
                                    backgroundColor: Colors.transparent,
                                    child: InteractiveViewer(
                                      child: Image.memory(
                                        base64Decode(widget.pseudoB64!),
                                        fit: BoxFit.contain,
                                      ),
                                    ),
                                  ),
                                );
                              }
                            },
                            child: safeImage(widget.pseudoB64, height: 160),
                          ),
                          info: 'Shows HER2 expression intensity in heatmap.',
                        ),
                      ],
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

/// ===== Gauge Widget =====
class GaugeWidget extends StatelessWidget {
  final double value;
  final Color accent;
  final Color darkText;

  const GaugeWidget(
      {super.key,
      required this.value,
      required this.accent,
      required this.darkText});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 160,
      height: 140,
      child: Stack(
        alignment: Alignment.center,
        children: [
          CustomPaint(
            size: const Size(160, 100),
            painter: GaugePainter(value: value, accent: accent),
          ),
          Positioned(
            top: 80,
            child: Text('${value.toInt()}%',
                style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: darkText)),
          ),
        ],
      ),
    );
  }
}

/// ===== Gauge Painter =====
class GaugePainter extends CustomPainter {
  final double value;
  final Color accent;

  GaugePainter({required this.value, required this.accent});

  @override
  void paint(Canvas canvas, Size size) {
    final rect = Rect.fromLTWH(0, 0, size.width, size.height * 2);
    final startAngle = 3.14;
    final sweepAngle = 3.14 * (value / 100);
    final paint = Paint()
      ..strokeWidth = 12
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;
    if (value > 75) {
      paint.color = accent;
    } else {
      paint.shader = LinearGradient(
        colors: [Color(0xFFFFC94A), Color(0xFFFFA726)],
      ).createShader(rect);
    }

    final bgPaint = Paint()
      ..color = Colors.grey[300]!
      ..strokeWidth = 12
      ..style = PaintingStyle.stroke;

    canvas.drawArc(rect, 3.14, 3.14, false, bgPaint);
    canvas.drawArc(rect, startAngle, sweepAngle, false, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}

