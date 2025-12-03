import 'package:flutter/material.dart';
import 'Result.dart';
import 'package:provider/provider.dart';
import '../widgets/sidebar.dart';
import '../theme_provider.dart';


class ProcessingScreen extends StatefulWidget {
  final Map<String, dynamic> result;
  final int processingTimeMs;

  const ProcessingScreen({
    super.key,
    required this.result,
    required this.processingTimeMs,
  });

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen> {
  String _activeRoute = '/processing';

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

  void _navigate(String route) {
    setState(() {
      _activeRoute = route;
    });
    Navigator.of(context).pushNamed(route);
  }
  double progress = 0.0;
  bool isDone = false;
  late int totalMs;

  @override
  void initState() {
  super.initState();
  debugPrint("ProcessingScreen result: ${widget.result}");
  totalMs = (widget.processingTimeMs > 0 ? widget.processingTimeMs : 5000).clamp(1000, 30000);
  simulateProcessing();
  }

  void simulateProcessing() async {
  final int steps = 20;
  final int stepMs = (totalMs / steps).ceil();

  for (int currentStep = 0; currentStep <= steps; currentStep++) {
    await Future.delayed(Duration(milliseconds: stepMs));
    if (!mounted) return;

    setState(() {
      progress = currentStep / steps;
      if (progress >= 1.0) {
        progress = 1.0;
        isDone = true;
      }
    });
  }
}

  @override
  Widget build(BuildContext context) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final bool isDark = themeProvider.themeMode == ThemeMode.dark;
    final Color darkBg = const Color(0xFF111827);
    final Color darkCard = const Color(0xFF1E293B);
    final Color darkText = const Color(0xFFE5E7EB);
    final Color darkTextSecondary = const Color(0xFF9CA3AF);
    final Color accent = const Color(0xFF4F7BFF);
    final Color lightBg = const Color(0xFFF9FAFB);
    final Color lightCard = Colors.white;
    final Color lightText = Colors.black;
    final Color lightTextSecondary = const Color(0xFF6B7280);

    final steps = [
      {
        'title': 'Preprocessing',
        'bullets': [
          'Resize the image to 224×224 pixels for correct reading.',
          'Adjust brightness and colors for accurate analysis.',
        ],
      },
      {
        'title': 'Model Inference',
        'bullets': [
          'The system converts the H&E image into a synthetic IHC version.',
          'It then analyzes the image to determine the HER2 score (0, 1+, 2+, or 3+).',
        ],
      },
      {
        'title': 'Grad-CAM',
        'bullets': [
          'Highlights regions the model focused on.',
          'Helps explain why the model made its prediction.',
        ],
      },
      {
        'title': 'Pseudo-Color Mapping',
        'bullets': [
          'Shows HER2 protein expression intensity.',
          'Uses different colors to highlight expression levels.',
        ],
      },
    ];

    // Determine current step based on progress
    int currentStep = 0;
    if (progress >= 0.75) {
      currentStep = 3;
    } else if (progress >= 0.5) {
      currentStep = 2;
    } else if (progress >= 0.25) {
      currentStep = 1;
    }

    return GestureDetector(
      behavior: HitTestBehavior.translucent,
      onTap: () {},
      child: Scaffold(
        backgroundColor: isDark ? darkBg : lightBg,
        body: SafeArea(
          child: Row(
            children: [
              Sidebar(
                onLogout: _logout,
                onNavigate: _navigate,
                activeRoute: _activeRoute,
                backgroundColor: isDark ? darkCard : lightCard,
                accentColor: accent,
                textColor: isDark ? darkText : lightText,
                secondaryTextColor: isDark ? darkTextSecondary : lightTextSecondary,
              ),
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    // Top box: steps bar
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 400,
                          padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 10),
                          decoration: BoxDecoration(
                            color: isDark ? Color(0xFF1E293B) : Colors.white,
                            borderRadius: BorderRadius.circular(18),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(isDark ? 0.18 : 0.08),
                                blurRadius: 14,
                                offset: Offset(0, 4),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisAlignment: MainAxisAlignment.spaceBetween,
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: List.generate(steps.length, (i) {
                              bool isDone = i < currentStep || progress == 1.0;
                              bool isCurrent = i == currentStep;
                              Color circleBorder = isDone || isCurrent ? Color(0xFF4F7BFF) : Color(0xFFD1D5DB);
                              Color textColor = (isDone || isCurrent) ? Colors.black : Color(0xFF9CA3AF);
                              return Container(
                                margin: EdgeInsets.symmetric(horizontal: 2),
                                padding: EdgeInsets.symmetric(horizontal: 5, vertical: 2),
                                decoration: BoxDecoration(
                                  color: isDark ? Color(0xFF1E293B) : Color(0xFFF6F8FC),
                                  borderRadius: BorderRadius.circular(8),
                                  boxShadow: [
                                    BoxShadow(
                                      color: Colors.black.withOpacity(0.07),
                                      blurRadius: 2,
                                      offset: Offset(0, 1),
                                    ),
                                  ],
                                ),
                                child: _StepWithPopover(
                                  stepIndex: i,
                                  isDone: isDone,
                                  circleBorder: circleBorder,
                                  textColor: textColor,
                                  step: steps[i],
                                  isDark: isDark,
                                ),
                              );
                            }),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 32),
                    // Bottom box: progress circle and title
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 400,
                          padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 18),
                          decoration: BoxDecoration(
                            color: isDark ? Color(0xFF1E293B) : Colors.white,
                            borderRadius: BorderRadius.circular(18),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(isDark ? 0.18 : 0.08),
                                blurRadius: 14,
                                offset: Offset(0, 4),
                              ),
                            ],
                          ),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            crossAxisAlignment: CrossAxisAlignment.center,
                            children: [
                              Text(
                                "Processing Image",
                                style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 8),
                              Text(
                                "Please wait while we process your image.",
                                style: TextStyle(fontSize: 14, color: isDark ? darkTextSecondary : lightTextSecondary),
                                textAlign: TextAlign.center,
                              ),
                              const SizedBox(height: 24),
                              SizedBox(
                                width: 220,
                                height: 220,
                                child: Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    CircularProgressIndicator(
                                      value: progress,
                                      strokeWidth: 8,
                                      backgroundColor: Colors.grey[300],
                                      valueColor: AlwaysStoppedAnimation<Color>(Color(0xFF4F7BFF)),
                                    ),
                                    Container(
                                      width: 90,
                                      height: 90,
                                      decoration: BoxDecoration(
                                        color: Colors.white,
                                        shape: BoxShape.circle,
                                        border: Border.all(color: Color(0xFF4F7BFF), width: 4),
                                      ),
                                      child: Center(
                                        child: Text(
                                          "${(progress * 100).toInt()}%",
                                          style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.black),
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              ),
                              const SizedBox(height: 24),
                              if (progress == 1.0)
                                ElevatedButton(
                                  onPressed: () {
                                    try {
                                      final result = widget.result;
                                      final her2Score = (result["pred_label"] ?? "-").toString();
                                      final gradcamLayers = (result["gradcam_layers"] is List)
                                        ? List<Map<String, dynamic>>.from(result["gradcam_layers"])
                                        : <Map<String, dynamic>>[];
                                      final pseudoB64 = (result["pseudo_b64"] ?? "").toString();
                                      final probsRaw = result["probs"];
                                      final probs = (probsRaw is Map)
                                        ? probsRaw.map((k, v) => MapEntry(k.toString(), v))
                                        : <String, dynamic>{};
                                      final origB64 = (result["orig_b64"] ?? "").toString();
                                      final confidence = num.tryParse(result["confidence"].toString()) ?? 0.0;
                                      final fileName = (result["file_name"] ?? '-').toString();
                                      final imageSize = (result["size"] ?? '-').toString();
                                      final analysisDate = (result["analysis_date"] ?? '-').toString();
                                      final probsChartB64 = (result["probs_chart_b64"] ?? "").toString();
                                      Navigator.of(context).pushReplacement(
                                        MaterialPageRoute(
                                            builder: (context) => ResultScreen(
                                              her2Score: her2Score,
                                              gradcamLayers: gradcamLayers,
                                              primaryGradcamB64: (result["primary_gradcam_b64"] ?? "").toString(),
                                              pseudoB64: pseudoB64,
                                              probs: probs,
                                              origB64: origB64,
                                              generated_b64: (result["generated_b64"] ?? "").toString(),
                                              confidence: confidence,
                                              fileName: fileName,
                                              imageSize: imageSize,
                                              analysisDate: analysisDate,
                                              probs_chart_b64: probsChartB64,
                                              isHneCheckbox: result["isHneCheckbox"] == true,
                                            ),
                                        ),
                                      );
                                    } catch (e, st) {
                                      debugPrint('Error navigating to ResultScreen: $e\n$st');
                                      ScaffoldMessenger.of(context).showSnackBar(
                                        SnackBar(content: Text('Eror showing results: $e')),
                                      );
                                    }
                                  },
                                  style: ElevatedButton.styleFrom(
                                    backgroundColor: accent,
                                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                    padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 32),
                                    elevation: 0,
                                  ),
                                  child: const Text("Show Results", style: TextStyle(fontSize: 15, color: Colors.white)),
                                ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }}
//
class _StepWithPopover extends StatefulWidget {
  final int stepIndex;
  final bool isDone;
  final Color circleBorder;
  final Color textColor;
  final Map<String, dynamic> step;

  final bool isDark;
  const _StepWithPopover({
    required this.stepIndex,
    required this.isDone,
    required this.circleBorder,
    required this.textColor,
    required this.step,
    required this.isDark,
  });

  @override
  State<_StepWithPopover> createState() => _StepWithPopoverState();
}

class _StepWithPopoverState extends State<_StepWithPopover> {
  OverlayEntry? _popoverEntry;

  void _showPopover() {
    final RenderBox box = context.findRenderObject() as RenderBox;
    final Offset position = box.localToGlobal(Offset.zero);
    _popoverEntry = OverlayEntry(
      builder: (context) => Positioned(
        left: position.dx,
        top: position.dy + box.size.height + 6,
        child: Material(
          color: Colors.transparent,
          child: Container(
            width: 320,
            decoration: BoxDecoration(
              color: widget.isDark ? Color(0xFF1E293B) : Colors.white,
              borderRadius: BorderRadius.circular(14),
              border: Border.all(color: const Color(0xFF4F7BFF), width: 1.5),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.13),
                  blurRadius: 16,
                  offset: const Offset(0, 4),
                ),
              ],
            ),
            padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  widget.step['title'] as String,
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: widget.isDark ? Color(0xFF4F7BFF) : Color(0xFF4F7BFF),
                  ),
                ),
                const SizedBox(height: 10),
                ...List.generate((widget.step['bullets'] as List).length, (b) {
                    final bullet = (widget.step['bullets'] as List)[b];
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 4),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text('-', style: TextStyle(fontSize: 15, color: Colors.black)),
                          const SizedBox(width: 7),
                          Expanded(
                            child: bullet is String
                                ? Text(
                                    bullet,
                                    style: TextStyle(fontSize: 12, color: widget.isDark ? Color(0xFFE5E7EB) : Color(0xFF222B45), height: 1.5),
                                  )
                                : bullet,
                          ),
                        ],
                      ),
                    );
                }),
                const SizedBox(height: 8),
                Align(
                  alignment: Alignment.centerRight,
                  child: GestureDetector(
                    onTap: _hidePopover,
                    child: Text(
                      "Close",
                      style: TextStyle(
                        color: Color(0xFF4F7BFF),
                        fontWeight: FontWeight.w600,
                        fontSize: 13,
                      ),
                    ),
                  ),
                ),
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
      child: Row(
        children: [
          Container(
            width: 14,
            height: 14,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              border: Border.all(color: widget.circleBorder, width: 2),
              color: widget.isDone ? widget.circleBorder : Colors.white,
            ),
            child: widget.isDone
                ? const Icon(Icons.check, size: 8, color: Colors.white)
                : null,
          ),
          const SizedBox(width: 6),
          Text(
            widget.step['title'] as String,
            style: TextStyle(fontSize: 6, fontWeight: FontWeight.w600, color: widget.textColor),
          ),
          const Icon(Icons.expand_more, size: 12, color: Color(0xFF9CA3AF)),
        ],
      ),
    );
  }
}
