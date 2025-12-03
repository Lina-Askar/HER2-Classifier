import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';
class _WavePainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    final path = Path();
    path.moveTo(0, size.height * 0.5);
    path.cubicTo(size.width * 0.25, size.height * 0.2, size.width * 0.75, size.height * 0.8, size.width, size.height * 0.5);
    canvas.drawPath(path, paint);
  
    path.reset();
    path.moveTo(0, size.height * 0.7);
    path.cubicTo(size.width * 0.25, size.height * 0.4, size.width * 0.75, size.height, size.width, size.height * 0.7);
    canvas.drawPath(path, paint);
  }
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class _DotsPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill;
    for (double y = 0; y < size.height; y += 10) {
      for (double x = 0; x < size.width; x += 10) {
        canvas.drawCircle(Offset(x, y), 2, paint);
      }
    }
  }
  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  bool _showAdminMsg = false;
  bool _obscurePassword = true;

  void _showImageDialog(BuildContext context, String imagePath) {
    showDialog(
      context: context,
      builder: (context) => Dialog(
        backgroundColor: Colors.transparent,
        child: GestureDetector(
          onTap: () => Navigator.of(context).pop(),
          child: InteractiveViewer(
            child: Container(
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(16),
                color: Colors.white,
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 12,
                  ),
                ],
              ),
              padding: const EdgeInsets.all(8),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(12),
                child: Image.asset(imagePath),
              ),
            ),
          ),
        ),
      ),
    );
  }

  late final List<Widget> _slides = [
   
    Center(
      child: Container(
        width: 270,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: const Color(0xFFF6F8FC),
          borderRadius: BorderRadius.circular(16),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 8,
              offset: Offset(0, 2),
            ),
          ],
        ),
        child: const SizedBox.shrink(), 
      ),
    ),
    
    Center(
      child: Container(
        width: 270,
        padding: const EdgeInsets.all(18),
        decoration: BoxDecoration(
          color: const Color(0xFFF6F8FC),
          borderRadius: BorderRadius.circular(16),
          boxShadow: const [
            BoxShadow(color: Colors.black12, blurRadius: 8, offset: Offset(0, 2)),
          ],
        ),
        child: const SizedBox.shrink(),
      ),
    ),
    
    Builder(
      builder: (context) => Center(
        child: Container(
          width: 270,
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: const Color(0xFFF6F8FC),
            borderRadius: BorderRadius.circular(16),
            boxShadow: const [
              BoxShadow(color: Colors.black12, blurRadius: 8, offset: Offset(0, 2)),
            ],
          ),
          child: const SizedBox.shrink(), 
        ),
      ),
    ),
  ];

  @override
  void initState() {
    super.initState();
  }

  @override
  void dispose() {
 
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
 
    final themeProvider = Theme.of(context);
    final bool isDark = themeProvider.brightness == Brightness.dark;
  final Color bgColor = const Color(0xFFF6F8FC);
  final Color cardColor = Colors.white;
    final Color textColor = isDark ? const Color(0xFFE5E7EB) : Colors.black;
    final Color secondaryTextColor = isDark ? const Color(0xFF9CA3AF) : const Color(0xFF8A8A8A);
  final Color accent = Colors.black;
  final Color inputBorderColor = Colors.black;
  final Color iconColor = Colors.black;
    return Scaffold(
      backgroundColor: bgColor,
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 800),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Container(
                width: 370,
                height: 420,
                decoration: BoxDecoration(
                  color: const Color(0xFF4F7BFF),
                  borderRadius: const BorderRadius.only(
                    topLeft: Radius.circular(16),
                    bottomLeft: Radius.circular(16),
                  ),
                ),
                child: Stack(
                  children: [
                    Positioned(
                      bottom: -70,
                      left: -60,
                      child: Container(
                        width: 220,
                        height: 220,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.white.withOpacity(0.09),
                        ),
                      ),
                    ),

                    Positioned(
                      top: -40,
                      right: -40,
                      child: Container(
                        width: 120,
                        height: 120,
                        decoration: BoxDecoration(
                          shape: BoxShape.circle,
                          color: Colors.white.withOpacity(0.13),
                        ),
                      ),
                    ),
                 
                    Positioned(
                      top: 60,
                      right: 18,
                      child: Opacity(
                        opacity: 0.12,
                        child: CustomPaint(
                          size: const Size(70, 50),
                          painter: _WavePainter(),
                        ),
                      ),
                    ),
               
                    Positioned(
                      bottom: 30,
                      right: 30,
                      child: Opacity(
                        opacity: 0.10,
                        child: CustomPaint(
                          size: const Size(60, 40),
                          painter: _DotsPainter(),
                        ),
                      ),
                    ),
                
                    Center(
                      child: Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 28),
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Text(
                              "Hello, welcome!",
                              textAlign: TextAlign.center,
                              style: const TextStyle(
                                color: Colors.white,
                                fontSize: 22,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                            const SizedBox(height: 18),
                            Text(
                              "This is the HER2 Classifier, an AI-powered system developed to assist pathologists in analyzing breast cancer tissue images. It automatically converts H&E slides into Synthetic IHC images and classifies HER2 expression levels (0 to 3+). With an accuracy of 94%, it provides reliable and consistent diagnostic support.",
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.white.withOpacity(0.85),
                                fontSize: 14,
                                height: 1.5,
                                fontWeight: FontWeight.normal,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              Container(
                width: 320,
                height: 420,
                decoration: BoxDecoration(
                  color: cardColor,
                  boxShadow: const [
                    BoxShadow(
                      color: Colors.black12,
                      blurRadius: 8,
                      offset: Offset(0, 2),
                    ),
                  ],
                  borderRadius: const BorderRadius.only(
                    topRight: Radius.circular(16),
                    bottomRight: Radius.circular(16),
                  ),
                ),
                child: Padding(
                  padding: const EdgeInsets.all(24),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      Center(
                        child: Image.asset(
                          'lib/assets/images/logo.png',
                          height: 55,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Text(
                        "Sign in",
                        textAlign: TextAlign.center,
                        style: TextStyle(
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                            color: accent),
                      ),
                    
                      const SizedBox(height: 22),
                      TextField(
                        controller: _usernameController,
                        style: const TextStyle(color: Colors.black),
                        decoration: InputDecoration(
                          labelText: "Username",
                          labelStyle: TextStyle(color: accent),
                          prefixIcon: Icon(Icons.person_outline, color: iconColor),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: BorderSide(color: accent, width: 1.2),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: BorderSide(color: accent, width: 2),
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      ),
                      const SizedBox(height: 14),
                      TextField(
                        controller: _passwordController,
                        obscureText: _obscurePassword,
                        style: const TextStyle(color: Colors.black),
                        decoration: InputDecoration(
                          labelText: "Password",
                          labelStyle: TextStyle(color: accent),
                          prefixIcon: Icon(Icons.lock_outline, color: iconColor),
                          suffixIcon: IconButton(
                              icon: Icon(_obscurePassword
                                  ? Icons.visibility_off
                                  : Icons.visibility, color: iconColor),
                              onPressed: () {
                                setState(() {
                                  _obscurePassword = !_obscurePassword;
                                });
                              },
                            ),
                          enabledBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: BorderSide(color: inputBorderColor, width: 1.2),
                          ),
                          focusedBorder: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                            borderSide: BorderSide(color: accent, width: 2),
                          ),
                          border: OutlineInputBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                        ),
                      ),
                      const SizedBox(height: 18),
                      SizedBox(
                        height: 38,
                        child: ElevatedButton(
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF587DFF),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(16),
                            ),
                            elevation: 2,
                            shadowColor:
                                const Color(0xFF587DFF).withOpacity(0.2),
                          ),
                          onPressed: () {
                            String username = _usernameController.text;
                            String password = _passwordController.text;
                            final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
                            const adminUsers = ["Lina", "Najla", "Farah", "Lama", "Kholoud"];
                            if (adminUsers.contains(username) && password == "12") {
                              themeProvider.setTheme(ThemeMode.light);
                              themeProvider.setAdmin(true);
                              SharedPreferences.getInstance().then((prefs) {
                                prefs.setString('current_username', username);
                                Navigator.pushReplacementNamed(context, '/upload');
                              });
                            } else {
                           
                              SharedPreferences.getInstance().then((prefs) {
                                final List<String>? doctorsJson = prefs.getStringList('doctors');
                                bool found = false;
                                if (doctorsJson != null) {
                                  for (var e in doctorsJson) {
                                    final user = Map<String, dynamic>.from(jsonDecode(e));
                                    if (user['username'] == username && user['password'] == password) {
                                      found = true;
                                      break;
                                    }
                                  }
                                }
                                if (found) {
                                  themeProvider.setAdmin(false);
                                  prefs.setString('current_username', username);
                                  Navigator.pushReplacementNamed(context, '/upload');
                                } else {
                                  themeProvider.setAdmin(false);
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                        content: Text(
                                            "Invalid username or password")),
                                  );
                                }
                              });
                            }
                          },
                          child: const Text(
                            "Sign in",
                            style:
                                TextStyle(color: Colors.white, fontSize: 15),
                          ),
                        ),
                      ),
                      const SizedBox(height: 8),
                      Center(
                        child: Column(
                          children: [
                            GestureDetector(
                              onTap: () {
                                setState(() {
                                  _showAdminMsg = true;
                                });
                              },
                              child: const Text(
                                "Forgot password?",
                                style: TextStyle(
                                  color: Color(0xFF587DFF),
                                  decoration: TextDecoration.underline,
                                  fontSize: 13,
                                ),
                              ),
                            ),
                            if (_showAdminMsg)
                              const Padding(
                                padding: EdgeInsets.only(top: 8.0),
                                child: Text(
                                  "Please contact the admin to reset your password.",
                                  style: TextStyle(
                                      color: Colors.red, fontSize: 12),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
