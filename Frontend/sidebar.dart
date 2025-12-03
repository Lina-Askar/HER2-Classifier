import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../theme_provider.dart';

// -------------------
// ExpandButton Widget
class ExpandButton extends StatelessWidget {
  final bool expanded;
  final VoidCallback onTap;
  const ExpandButton({super.key, required this.expanded, required this.onTap});
  @override
  Widget build(BuildContext context) {
    return Positioned(
      top: 18,
      right: -18,
      child: GestureDetector(
        onTap: onTap,
        child: Container(
          width: 36,
          height: 36,
          decoration: BoxDecoration(
            color: const Color(0xFF587DFF),
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.08),
                blurRadius: 4,
                offset: const Offset(2, 0),
              ),
            ],
          ),
          child: Icon(
            expanded ? Icons.arrow_back_ios_new : Icons.arrow_forward_ios,
            color: Colors.white,
            size: 20,
          ),
        ),
  )
  );
  }
}

// -------------------
// Sidebar Widget
class Sidebar extends StatefulWidget {
  final VoidCallback onLogout;
  final Function(String route) onNavigate;
  final String activeRoute;
  final Color? backgroundColor;
  final Color? accentColor;
  final Color? textColor;
  final Color? secondaryTextColor;

  const Sidebar({
    super.key,
    required this.onLogout,
    required this.onNavigate,
    required this.activeRoute,
    this.backgroundColor,
    this.accentColor,
    this.textColor,
    this.secondaryTextColor,
  });

  @override
  State<Sidebar> createState() => _SidebarState();
}

// -------------------
// Sidebar State
class _SidebarState extends State<Sidebar> {
  bool _expanded = false;
  @override
  Widget build(BuildContext context) {
    final Color accent = widget.accentColor ?? const Color(0xFF4F7BFF);
    final Color bg = widget.backgroundColor ?? const Color(0xFFF6F8FC);
    final Color card = widget.backgroundColor ?? Colors.white;
    final Color text = widget.textColor ?? Colors.black;
    final Color textSecondary = widget.secondaryTextColor ?? const Color(0xFF6B7280);
    final themeProvider = Provider.of<ThemeProvider>(context);
    final bool isDark = themeProvider.themeMode == ThemeMode.dark;

    return Container(
      color: bg,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 250),
        curve: Curves.ease,
        width: _expanded ? 220 : 60,
        decoration: BoxDecoration(
          color: card,
          boxShadow: [
            BoxShadow(
              color: isDark ? Colors.black.withOpacity(0.18) : Colors.grey.withOpacity(0.10),
              blurRadius: 12,
              offset: const Offset(2, 0),
            ),
          ],
          borderRadius: BorderRadius.circular(18),
        ),
        child: Column(
          children: [
            const SizedBox(height: 24),
            // Logo centered and clickable to expand/collapse
            Center(
              child: GestureDetector(
                onTap: () {
                  setState(() {
                    _expanded = !_expanded;
                  });
                },
                child: Image.asset(
                  isDark ? 'lib/assets/images/darklogo.png' : 'lib/assets/images/logo.png',
                  height: 36,
                  width: 36,
                ),
              ),
            ),
            const SizedBox(height: 24),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  if (_expanded) ...[
                    Tooltip(
                      message: 'Analyze Image',
                      child: ListTile(
                        leading: Icon(Icons.upload_file, color: accent),
                        title: Text('Analyze Image', style: TextStyle( color: text)),
                        selected: widget.activeRoute == '/upload',
                        selectedTileColor: accent.withOpacity(0.12),
                        onTap: () => widget.onNavigate('/upload'),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16),
                        minLeadingWidth: 0,
                      ),
                    ),
                    Tooltip(
                      message: 'History',
                      child: ListTile(
                        leading: Icon(Icons.history, color: accent),
                        title: Text('History', style: TextStyle(color: text)),
                        selected: widget.activeRoute == '/history',
                        selectedTileColor: accent.withOpacity(0.12),
                        onTap: () => widget.onNavigate('/history'),
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16),
                        minLeadingWidth: 0,
                      ),
                    ),
                    if (themeProvider.isAdmin)
                      Tooltip(
                        message: 'Settings',
                        child: ListTile(
                          leading: Icon(Icons.settings, color: accent),
                          title: Text('Settings', style: TextStyle(color: text)),
                          selected: widget.activeRoute == '/admin_settings_screen',
                          selectedTileColor: accent.withOpacity(0.12),
                          onTap: () => widget.onNavigate('/admin_settings_screen'),
                          contentPadding: const EdgeInsets.symmetric(horizontal: 16),
                          minLeadingWidth: 0,
                        ),
                      ),
                  ] else ...[
                    Tooltip(
                      message: 'Analyze Image',
                      child: IconButton(
                        icon: Icon(Icons.upload_file, color: accent),
                        onPressed: () => widget.onNavigate('/upload'),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Tooltip(
                      message: 'History',
                      child: IconButton(
                        icon: Icon(Icons.history, color: accent),
                        onPressed: () => widget.onNavigate('/history'),
                      ),
                    ),
                    const SizedBox(height: 16),
                    if (themeProvider.isAdmin)
                      Tooltip(
                        message: 'Settings',
                        child: IconButton(
                          icon: Icon(Icons.settings, color: accent),
                          onPressed: () => widget.onNavigate('/admin_settings_screen'),
                        ),
                      ),
                    if (themeProvider.isAdmin)
                      const SizedBox(height: 16),
                    const SizedBox(height: 16),
                  ],
                  const Spacer(),
                  if (_expanded)
                    Padding(
                      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.wb_sunny, color: isDark ? Colors.grey[400] : accent, size: 22),
                          const SizedBox(width: 8),
                          GestureDetector(
                            onTap: () => themeProvider.toggleTheme(),
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 200),
                              width: 56,
                              height: 32,
                              padding: const EdgeInsets.symmetric(horizontal: 4),
                              decoration: BoxDecoration(
                                color: isDark ? accent.withOpacity(0.7) : accent.withOpacity(0.2),
                                borderRadius: BorderRadius.circular(20),
                              ),
                              child: Stack(
                                alignment: Alignment.center,
                                children: [
                                  AnimatedAlign(
                                    duration: const Duration(milliseconds: 200),
                                    alignment: isDark ? Alignment.centerRight : Alignment.centerLeft,
                                    child: Container(
                                      width: 24,
                                      height: 24,
                                      decoration: BoxDecoration(
                                        color: Colors.white,
                                        shape: BoxShape.circle,
                                        boxShadow: [
                                          BoxShadow(
                                            color: Colors.black.withOpacity(0.08),
                                            blurRadius: 2,
                                            offset: const Offset(0, 1),
                                          ),
                                        ],
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Icon(
                            Icons.nightlight_round,
                            color: isDark ? accent : Colors.grey[400],
                            size: 22,
                          ),
                        ],
                      ),
                    )
                  else
                    Padding(
                      padding: const EdgeInsets.only(bottom: 8.0),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.wb_sunny, color: isDark ? Colors.grey[400] : accent, size: 22),
                          const SizedBox(height: 6),
                          GestureDetector(
                            onTap: () => themeProvider.toggleTheme(),
                            child: AnimatedContainer(
                              duration: const Duration(milliseconds: 200),
                              width: 36,
                              height: 36,
                              decoration: BoxDecoration(
                                color: isDark ? accent.withOpacity(0.7) : accent.withOpacity(0.2),
                                shape: BoxShape.circle,
                              ),
                              child: Icon(
                                isDark ? Icons.nightlight_round : Icons.wb_sunny,
                                color: Colors.white,
                                size: 22,
                              ),
                            ),
                          ),
                          const SizedBox(height: 6),
                          Icon(
                            Icons.nightlight_round,
                            color: isDark ? accent : Colors.grey[400],
                            size: 22,
                          ),
                        ],
                      ),
                    ),
                  Padding(
                    padding: const EdgeInsets.only(bottom: 18.0),
                    child: Center(
                      child: Tooltip(
                        message: 'Sign Out',
                        child: GestureDetector(
                          onTap: widget.onLogout,
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                Icons.logout,
                                color: accent,
                                size: 32,
                              ),
                              if (_expanded)
                                const SizedBox(width: 8),
                              if (_expanded)
                                Text('Logout', style: TextStyle(fontSize: 18, color: textSecondary)),
                            ],
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

