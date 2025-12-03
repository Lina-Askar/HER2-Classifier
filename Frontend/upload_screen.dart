import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;
import 'dart:io';
import 'package:http/io_client.dart';
import '../widgets/sidebar.dart';
import 'package:dotted_border/dotted_border.dart';
import '../theme_provider.dart';
import 'Processing.dart';
import 'dart:convert';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:image/image.dart' as img;

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  // ------------------ State Fields ------------------
  XFile? _selectedImage;
  String? _imageSizeError;
  bool _isProcessing = false;

  bool _isHE = false;
  bool _showHEPopover = false;
  bool _showIHCPopover = false;
  final LayerLink _heLink = LayerLink();
  final LayerLink _ihcLink = LayerLink();

  String? _apiBase;
  String _activeRoute = '/upload';

  // ------------------ Init / Dispose ------------------
  @override
  void initState() {
    super.initState();
    _loadApiBase();
  }

  Future<void> _loadApiBase() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      _apiBase = prefs.getString('ihcnet_api_link');
    });
  }

  @override
  void dispose() {
    _selectedImage = null;
    _imageSizeError = null;
    _isProcessing = false;
    _isHE = false;
    super.dispose();
  }

  // ------------------ Pick / Remove ------------------
  void _pickImage() async {
    final ImagePicker picker = ImagePicker();
    final XFile? image = await picker.pickImage(source: ImageSource.gallery);
    if (!mounted) return;
    if (image != null) {
      setState(() {
        _selectedImage = image;
        _imageSizeError = null;
      });
    }
  }

  void _removeImage() {
    setState(() {
      _selectedImage = null;
      _imageSizeError = null;
      _isProcessing = false;
    });
  }

  // ------------------ Logout / Navigate ------------------
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

  // ------------------ Process Image ------------------
  Future<void> _processImage() async {
    final prefs = await SharedPreferences.getInstance();
    String apiUrl = '';
    if (_isHE) {
      apiUrl = (prefs.getString('PSPStain_api_link') ?? '').trim();
      if (!apiUrl.endsWith('/GANpredict')) {
        apiUrl = apiUrl.replaceAll(RegExp(r'/+$'), '') + '/GANpredict';
      }
    } else {
      apiUrl = (prefs.getString('ihcnet_api_link') ?? '').trim();
      if (!apiUrl.endsWith('/predict')) {
        apiUrl = apiUrl.replaceAll(RegExp(r'/+$'), '') + '/predict';
      }
    }
    debugPrint('API URL used: $apiUrl');

    if (_isProcessing) {
      debugPrint(' Already processing — skipping.');
      return;
    }

    if (_selectedImage == null) {
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('No Image'),
          content: const Text('Please select an image before processing.'),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
        ),
      );
      return;
    }

    setState(() {
      _isProcessing = true;
      _imageSizeError = null;
    });
    if (apiUrl.isEmpty) {
      if (!mounted) return;
      showDialog(
        context: context,
        builder: (context) => AlertDialog(
          title: const Text('API Link Missing'),
          content: const Text('Please set the API link in Admin Settings.'),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
        ),
      );
      setState(() {
        _isProcessing = false;
      });
      return;
    }
    bool dialogOpen = false;
    IOClient? httpClient;

    try {
    
      final bytes = await File(_selectedImage!.path).readAsBytes();
      final decodedImg = img.decodeImage(bytes);
      if (decodedImg == null) {
        throw Exception('Unsupported image format or corrupted image');
      }
      debugPrint(' Image size: ${decodedImg.width}x${decodedImg.height}');
      if (decodedImg.width < 224 || decodedImg.height < 224) {
        setState(() {
          _imageSizeError = 'Image size must be at least 224×224 px';
          _isProcessing = false;
        });
        return;
      }

      String tempFileName = _selectedImage!.name;
      

      if (!mounted) return;
      showDialog(
        context: context,
        barrierDismissible: false,
        builder: (context) => const Center(child: CircularProgressIndicator()),
      );
      dialogOpen = true;

      final ioc = HttpClient()..badCertificateCallback = (X509Certificate cert, String host, int port) => true;
      httpClient = IOClient(ioc);

      final uri = Uri.parse(apiUrl);
      final request = http.MultipartRequest('POST', uri);
      request.files.add(await http.MultipartFile.fromPath('file', _selectedImage!.path));
      request.fields['is_he'] = _isHE.toString();

      debugPrint(' Sending request to $uri');
      final streamedResponse = await httpClient.send(request).timeout(const Duration(seconds: 120));
      final response = await http.Response.fromStream(streamedResponse);

      debugPrint(' Response status: ${response.statusCode}');
      if (response.statusCode != 200) {
        throw Exception('API Error: ${response.statusCode}');
      }

      final apiResult = jsonDecode(response.body);
        final genB64 = (apiResult['generated_b64'] ?? '').toString();
        debugPrint(' [generated_b64] from backend: ${genB64.isNotEmpty ? (genB64.length > 50 ? genB64.substring(0, 50) : genB64) : 'NULL'}');
       
      debugPrint(' API result parsed');

       
      final her2Score = apiResult['pred_label'] ?? '-';
      final gradcamLayers = apiResult['gradcam_layers'] ?? [];
      final pseudoB64 = apiResult['pseudo_b64'] ?? '';
      final probs = (apiResult['probs'] is Map) ? Map<String, dynamic>.from(apiResult['probs']) : <String, dynamic>{};
      final origB64 = apiResult['orig_b64'] ?? '';
      final confidence = apiResult['confidence'] ?? 0.0;
      final probsChartB64 = apiResult['probs_chart_b64'] ?? '';
      final primaryGradcam = apiResult['primary_gradcam_b64'] ?? '';

      if (dialogOpen && mounted) {
        Navigator.of(context, rootNavigator: true).pop();
        dialogOpen = false;
      }

      if (!mounted) return;
      debugPrint(' [generated_b64] sent to ProcessingScreen: ${genB64.isNotEmpty ? (genB64.length > 50 ? genB64.substring(0, 50) : genB64) : 'NULL'}');
      final result = await Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ProcessingScreen(
            result: {
              'pred_label': her2Score,
              'gradcam_layers': gradcamLayers,
              'primary_gradcam_b64': primaryGradcam,
              'pseudo_b64': pseudoB64,
              'probs': probs,
              'orig_b64': origB64,
              'generated_b64': apiResult['generated_b64'] ?? '',
              'origResizedB64': apiResult['origResizedB64'] ?? '',
              'confidence': confidence,
              'file_name': tempFileName,
              'size': '${decodedImg.width}x${decodedImg.height}',
              'analysis_date': DateTime.now().toIso8601String(),
              'probs_chart_b64': probsChartB64,
              'isHneCheckbox': _isHE,
            },
            processingTimeMs: 3000,
          ),
        ),
      );

      if (!mounted) return;
      if (result == true) {
        setState(() {
          _selectedImage = null;
          _imageSizeError = null;
        });
      }
    } catch (e, st) {
      debugPrint(' _processImage exception: $e\n$st');
      if (mounted) {
        try {
          if (dialogOpen) {
            Navigator.of(context, rootNavigator: true).pop();
            dialogOpen = false;
          }
        } catch (_) {}
        showDialog(
          context: context,
          builder: (context) => AlertDialog(
            title: const Text('Error'),
            content: Text('Failed to process image.\n$e'),
            actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
          ),
        );
      }
    } finally {
      try {
        httpClient?.close();
      } catch (_) {}
      if (mounted) {
        setState(() {
          _isProcessing = false;
        });
      }
      debugPrint(' _processImage finished, cleaned up.');
    }
  }

  // ------------------ Popovers ------------------
  void _closePopovers() {
    setState(() {
      _showHEPopover = false;
      _showIHCPopover = false;
    });
  }

  // ------------------ Build ------------------
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

    return GestureDetector(
      behavior: HitTestBehavior.translucent,
      onTap: _closePopovers,
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
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      const SizedBox(height: 40),
                      Center(
                        child: ConstrainedBox(
                          constraints: const BoxConstraints(maxWidth: 400),
                          child: Container(
                            padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 24),
                            decoration: BoxDecoration(
                              color: isDark ? darkCard : lightCard,
                              borderRadius: BorderRadius.circular(18),
                              boxShadow: [
                                BoxShadow(
                                  color: Colors.black.withOpacity(isDark ? 0.18 : 0.04),
                                  blurRadius: 16,
                                  offset: const Offset(0, 4),
                                ),
                              ],
                              border: Border.all(color: accent.withOpacity(0.3), width: 2),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.center,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Stack(
                                  alignment: Alignment.center,
                                  children: [
                                    Center(
                                      child: Text(
                                        'Start Analysis – Upload IHC Image',
                                        style: TextStyle(
                                          fontSize: 19,
                                          fontWeight: FontWeight.bold,
                                          color: isDark ? darkText : lightText,
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                                const SizedBox(height: 16),

                                // Drag & Drop + Tap
                                DragTarget<XFile>(
                                  onAcceptWithDetails: (details) {
                                    setState(() {
                                      _selectedImage = details.data;
                                      _imageSizeError = null;
                                    });
                                  },
                                  builder: (context, candidateData, rejectedData) {
                                    final hovered = candidateData.isNotEmpty;
                                    return GestureDetector(
                                      onTap: _pickImage,
                                      child: DottedBorder(
                                        color: accent,
                                        borderType: BorderType.RRect,
                                        radius: const Radius.circular(12),
                                        dashPattern: const [8, 4],
                                        strokeWidth: 2,
                                        child: Container(
                                          width: double.infinity,
                                          height: 220,
                                          alignment: Alignment.center,
                                          color: hovered ? accent.withOpacity(0.08) : Colors.transparent,
                                          child: _selectedImage == null
                                              ? Column(
                                                  mainAxisAlignment: MainAxisAlignment.center,
                                                  children: [
                                                    Icon(Icons.cloud_upload, size: 48, color: accent),
                                                    const SizedBox(height: 12),
                                                    Text(
                                                      'Drag & drop to upload or browse',
                                                      style: TextStyle(
                                                        fontSize: 16,
                                                        color: isDark ? darkTextSecondary : Colors.grey[700],
                                                      ),
                                                    ),
                                                    const SizedBox(height: 8),
                                                    Column(
                                                      children: [
                                                        RichText(
                                                          text: TextSpan(
                                                            style: TextStyle(
                                                              fontSize: 14,
                                                              color: isDark ? darkTextSecondary : lightTextSecondary,
                                                            ),
                                                            children: [
                                                              const TextSpan(text: 'Accepted types: '),
                                                              TextSpan(
                                                                text: 'WSI, TMA, Patch',
                                                                style: TextStyle(color: accent, fontWeight: FontWeight.w600),
                                                              ),
                                                            ],
                                                          ),
                                                        ),
                                                        const SizedBox(height: 2),
                                                        RichText(
                                                          text: TextSpan(
                                                            style: TextStyle(
                                                              fontSize: 14,
                                                              color: isDark ? darkTextSecondary : lightTextSecondary,
                                                            ),
                                                            children: [
                                                              const TextSpan(text: 'Formats: '),
                                                              TextSpan(
                                                                text: 'PNG, JPG, JPEG',
                                                                style: TextStyle(color: accent, fontWeight: FontWeight.w600),
                                                              ),
                                                            ],
                                                          ),
                                                        ),
                                                      ],
                                                    ),
                                                    const SizedBox(height: 8),
                                                    Text(
                                                      'Min: 224×224 px',
                                                      style: TextStyle(fontSize: 13, color: isDark ? darkTextSecondary : lightTextSecondary),
                                                    ),
                                                    const SizedBox(height: 2),
                                                    Text(
                                                      'Recommended max: 4096×4096 px',
                                                      style: TextStyle(fontSize: 12, color: isDark ? darkTextSecondary : lightTextSecondary),
                                                    ),
                                                  ],
                                                )
                                              : Image.file(
                                                  File(_selectedImage!.path),
                                                  fit: BoxFit.contain,
                                                  height: 180, 
                                                ),
                                        ),
                                      ));
                                    },
                                  ),

                                const SizedBox(height: 20),

                                // H&E Checkbox + Popover 
                                Row(
                                  crossAxisAlignment: CrossAxisAlignment.center,
                                  children: [
                                    Checkbox(
                                      value: _isHE,
                                      onChanged: (val) => setState(() => _isHE = val ?? false),
                                      activeColor: const Color(0xFF4F7BFF),
                                      checkColor: Colors.white,
                                      visualDensity: VisualDensity.compact,
                                      materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                                    ),
                                    const SizedBox(width: 8),
                                    Expanded(
                                      child: Row(
                                        crossAxisAlignment: CrossAxisAlignment.center,
                                        children: [
                                          const Flexible(
                                            child: Padding(
                                              padding: EdgeInsets.only(left: 4.0, right: 2.0),
                                              child: Text(
                                                'Input image type: H&E',
                                                style: TextStyle(
                                                  fontSize: 13,
                                                  height: 1.5,
                                                  color: Color(0xFF4F7BFF),
                                                ),
                                                maxLines: 2,
                                                overflow: TextOverflow.visible,
                                                textAlign: TextAlign.left,
                                              ),
                                            ),
                                          ),
                                          CompositedTransformTarget(
                                            link: _heLink,
                                            child: GestureDetector(
                                              onTap: () {
                                                setState(() {
                                                  _showHEPopover = !_showHEPopover;
                                                  _showIHCPopover = false; 
                                                });
                                              },
                                              child: const Padding(
                                                padding: EdgeInsets.only(left: 2.0),
                                                child: Icon(Icons.help_outline, color: Color(0xFF4F7BFF), size: 18),
                                              ),
                                            ),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ],
                                ),

                                const SizedBox(height: 24),

                                // Buttons
                                Row(
                                  children: [
                                    Expanded(
                                      child: OutlinedButton(
                                        onPressed: _selectedImage != null ? _removeImage : null,
                                        style: OutlinedButton.styleFrom(
                                          side: const BorderSide(color: Color(0xFFE74C3C), width: 2),
                                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                          padding: const EdgeInsets.symmetric(vertical: 14),
                                          foregroundColor: const Color(0xFFE74C3C),
                                          backgroundColor: Colors.transparent,
                                        ),
                                        child: const Text('Remove', style: TextStyle(fontSize: 16, color: Color(0xFFE74C3C))),
                                      ),
                                    ),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: ElevatedButton(
                                        onPressed: (_selectedImage != null && !_isProcessing) ? _processImage : null,
                                        style: ElevatedButton.styleFrom(
                                          backgroundColor: _selectedImage != null ? const Color(0xFF4F7BFF) : const Color(0xFFB3C7FF),
                                          disabledBackgroundColor: const Color(0xFFB3C7FF),
                                          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                          padding: const EdgeInsets.symmetric(vertical: 14),
                                          elevation: 0,
                                        ).copyWith(
                                          overlayColor: WidgetStateProperty.resolveWith<Color?>(
                                            (Set<WidgetState> states) {
                                              if (states.contains(WidgetState.hovered)) {
                                                return const Color(0xFF7DC9FF);
                                              }
                                              if (states.contains(WidgetState.pressed)) {
                                                return const Color(0xFF3B5EDC);
                                              }
                                              return null;
                                            },
                                          ),
                                        ),
                                        child: const Text('Process', style: TextStyle(fontSize: 16, color: Colors.white)),
                                      ),
                                    ),
                                  ],
                                ),

                                if (_imageSizeError != null)
                                  Padding(
                                    padding: const EdgeInsets.only(top: 10.0),
                                    child: Text(
                                      _imageSizeError!,
                                      style: const TextStyle(color: Colors.red, fontSize: 14, fontWeight: FontWeight.w600),
                                    ),
                                  ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),

        floatingActionButton: Stack(
          children: [
            if (_showHEPopover)
              Positioned.fill(
                child: _buildPopover(
                  context,
                  _heLink,
                  '',
                  'Check this box if your image is H&E.\nThe system will convert it to IHC automatically.',
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ------------------ Popover Builder ------------------
  Widget _buildPopover(BuildContext context, LayerLink link, String imagePath, String text) {
    final themeProvider = Provider.of<ThemeProvider>(context);
    final bool isDark = themeProvider.themeMode == ThemeMode.dark;

    final Color accent = const Color(0xFF4F7BFF);
    final Color darkCard = const Color(0xFF1E293B);
    final Color lightCard = Colors.white;
    final Color popoverBg = isDark ? darkCard : lightCard;
    final Color popoverText = isDark ? const Color(0xFFE5E7EB) : const Color(0xFF4F7BFF);

    String title = '';
    String desc = '';
    String caption = '';

    if (imagePath.contains('pix')) {
      title = 'PSPStain(Image Conversion Model)';
      desc = 'A deep learning model that converts regular H&E images into IHC-like images.';
      caption = 'It helps visualize protein expression digitally — no actual staining required.';
    } else if (imagePath.contains('ihcnet')) {
      title = 'IHCNet (HER2 Classification Model)';
      desc = 'A neural network that analyzes IHC images to automatically score HER2 expression (0, 1+, 2+, 3+).';
      caption = 'It supports pathologists by providing consistent and objective HER2 grading.';
    }

    return Stack(
      children: [
        GestureDetector(
          behavior: HitTestBehavior.translucent,
          onTap: _closePopovers,
          child: Container(),
        ),
        CompositedTransformFollower(
          link: link,
          showWhenUnlinked: false,
          offset: const Offset(0, 28),
          child: Material(
            color: Colors.transparent,
            child: Container(
              width: 240,
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: popoverBg,
                borderRadius: BorderRadius.circular(12),
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.12),
                    blurRadius: 16,
                    offset: const Offset(0, 4),
                  ),
                ],
                border: Border.all(color: accent, width: 1.5),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  if (imagePath.isEmpty) ...[
                    Text(
                      text,
                      style: TextStyle(fontSize: 12, color: popoverText, height: 1.5),
                      textAlign: TextAlign.center,
                    ),
                  ] else ...[
                    Text(
                      title,
                      style: TextStyle(fontSize: 9, fontWeight: FontWeight.bold, color: accent),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(
                          maxHeight: 180,
                          minHeight: 80,
                        ),
                        child: Image.asset(
                          imagePath,
                          width: double.infinity,
                          fit: BoxFit.contain,
                          errorBuilder: (context, error, stackTrace) => const SizedBox.shrink(),
                        ),
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      desc,
                      style: TextStyle(fontSize: 10, color: popoverText, height: 1.5),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 6),
                    Text(
                      caption,
                      style: TextStyle(fontSize: 9, color: popoverText, height: 1.4),
                      textAlign: TextAlign.center,
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}
