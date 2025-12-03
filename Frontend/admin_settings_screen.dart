import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

import '../theme_provider.dart';
import '../widgets/sidebar.dart';

class AdminSettingsScreen extends StatefulWidget {
  const AdminSettingsScreen({super.key});

  @override
  State<AdminSettingsScreen> createState() => _AdminSettingsScreenState();
}

class _AdminSettingsScreenState extends State<AdminSettingsScreen> {
  @override
  void initState() {
    super.initState();
    _loadDoctors();
    _loadModelConfig();
  }

  Future<void> _loadModelConfig() async {
    final prefs = await SharedPreferences.getInstance();
    setState(() {
      PSPStainLinkController.text = (prefs.getString('PSPStain_api_link') ?? '').trim();
      ihcnetLinkController.text = (prefs.getString('ihcnet_api_link') ?? '').trim().toLowerCase();
      PSPStainPthController.text = (prefs.getString('PSPStain_pth_path') ?? '').trim();
      ihcnetPthController.text = (prefs.getString('ihcnet_pth_path') ?? '').trim();
    });
  }

  Future<void> _loadDoctors() async {
    final prefs = await SharedPreferences.getInstance();
    final List<String>? doctorsJson = prefs.getStringList('doctors');
    if (doctorsJson != null) {
      setState(() {
        doctors = doctorsJson.map((e) => Map<String, String>.from(Map<String, dynamic>.from(jsonDecode(e)))).toList();
      });
    }
  }

  Future<void> _saveDoctors() async {
    final prefs = await SharedPreferences.getInstance();
    final List<String> doctorsJson = doctors.map((e) => jsonEncode(e)).toList();
    await prefs.setStringList('doctors', doctorsJson);
  }
  // ================= Model Config Controllers =================
  final TextEditingController PSPStainLinkController = TextEditingController();
  final TextEditingController ihcnetLinkController = TextEditingController();
  final TextEditingController PSPStainPthController = TextEditingController();
  final TextEditingController ihcnetPthController = TextEditingController();

  // ================= user Management =================
  List<Map<String, String>> doctors = [];
  final TextEditingController usernameController = TextEditingController();
  final TextEditingController passwordController = TextEditingController();

  int? editingIndex;

  String? _searchText;
  int _currentPage = 0;

  // ================= Methods =================
  void saveModelConfig() {
    SharedPreferences.getInstance().then((prefs) {
      prefs.setString('ihcnet_api_link', ihcnetLinkController.text.trim().toLowerCase());
      prefs.setString('PSPStain_api_link', PSPStainLinkController.text.trim());
      prefs.setString('ihcnet_pth_path', ihcnetPthController.text.trim());
      prefs.setString('PSPStain_pth_path', PSPStainPthController.text.trim());
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Model configuration saved successfully!')),
      );
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
              Navigator.pop(context);
              Navigator.of(context).pushReplacementNamed('/');
            },
            child: const Text('Log Out', style: TextStyle(color: Colors.white)),
          ),
        ],
      ),
    );
  }
  
  Future<void> addOrUpdateDoctor() async {
    if (usernameController.text.isEmpty || passwordController.text.isEmpty) return;

    if (editingIndex != null) {
      doctors[editingIndex!] = {
        'username': usernameController.text,
        'password': passwordController.text,
      };
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Doctor updated successfully!')),
      );
    } else {
      doctors.add({
        'username': usernameController.text,
        'password': passwordController.text,
      });
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('user added successfully!')),
      );
    }

    usernameController.clear();
    passwordController.clear();
    editingIndex = null;
    await _saveDoctors();
    setState(() {});
  }

  void editDoctor(int index) {
    setState(() {
      editingIndex = index;
      usernameController.text = doctors[index]['username']!;
      passwordController.text = doctors[index]['password']!;
    });
  }

  Future<void> deleteDoctor(int index) async {
    setState(() {
      doctors.removeAt(index);
    });
    await _saveDoctors();
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('user deleted successfully!')),
    );
  }

  // ================= UI =================
  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final themeProvider = Provider.of<ThemeProvider>(context);
    final bool isDark = themeProvider.themeMode == ThemeMode.dark;
    final Color accent = const Color(0xFF4F7BFF);
    final Color cardColor = isDark ? const Color(0xFF1E293B) : Colors.white;
    final Color textColor = isDark ? const Color(0xFFE5E7EB) : Colors.black;
    final Color textSecondary = isDark ? const Color(0xFF9CA3AF) : const Color(0xFF6B7280);
    final Color scaffoldBg = isDark ? const Color(0xFF111827) : const Color(0xFFF8F9FB);
    final Color containerBg = isDark ? const Color(0xFF1E293B) : Colors.white;

    return Scaffold(
      backgroundColor: scaffoldBg,
      body: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Sidebar(
            onLogout: _logout,
            onNavigate: (route) => Navigator.of(context).pushReplacementNamed(route),
            activeRoute: '/admin_settings_screen',
            backgroundColor: cardColor,
            accentColor: accent,
            textColor: textColor,
            secondaryTextColor: textSecondary,
          ),
          Expanded(
            child: SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const SizedBox(height: 16),
                  Text(
                    "Admin Settings",
                    style: theme.textTheme.titleLarge?.copyWith(
                      fontSize: 28,
                      fontWeight: FontWeight.bold,
                      color: textColor,
                    ),
                  ),
                  const SizedBox(height: 24),
                  // ================= Model Config =================
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: containerBg,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: isDark ? Colors.black12 : Colors.grey.withOpacity(0.1),
                          blurRadius: 6,
                          offset: const Offset(0, 2),
                        )
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "Model Configuration",
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: textColor,
                          ),
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller: PSPStainLinkController,
                                decoration: const InputDecoration(
                                  labelText: "PSPStain Model Link",
                                  border: OutlineInputBorder(),
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: TextField(
                                controller: ihcnetLinkController,
                                decoration: const InputDecoration(
                                  labelText: "IHCNet Model Link",
                                  border: OutlineInputBorder(),
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 12),
                        Row(
                          children: [
                            Expanded(
                              child: TextField(
                                controller: PSPStainPthController,
                                decoration: InputDecoration(
                                  labelText: "PSPStain .pth File Path",
                                  border: const OutlineInputBorder(),
                                  suffixIcon: IconButton(
                                    icon: const Icon(Icons.upload_file),
                                    onPressed: () async {
                                      final result = await FilePicker.platform.pickFiles(type: FileType.any);
                                      if (result != null && result.files.isNotEmpty) {
                                        PSPStainPthController.text = result.files.single.path ?? '';
                                      }
                                    },
                                  ),
                                ),
                              ),
                            ),
                            const SizedBox(width: 16),
                            Expanded(
                              child: TextField(
                                controller: ihcnetPthController,
                                decoration: InputDecoration(
                                  labelText: "IHCNet .pth File Path",
                                  border: const OutlineInputBorder(),
                                  suffixIcon: IconButton(
                                    icon: const Icon(Icons.upload_file),
                                    onPressed: () async {
                                      final result = await FilePicker.platform.pickFiles(type: FileType.any);
                                      if (result != null && result.files.isNotEmpty) {
                                        ihcnetPthController.text = result.files.single.path ?? '';
                                      }
                                    },
                                  ),
                                ),
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Align(
                          alignment: Alignment.centerRight,
                          child: ElevatedButton.icon(
                            onPressed: saveModelConfig,
                            icon: const Icon(Icons.save),
                            label: const Text("Save Changes"),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(0xFF4F7BFF),
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(
                                horizontal: 20, vertical: 12),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 30),

                  // ================= Doctors Management =================
                  Container(
                    padding: const EdgeInsets.all(20),
                    decoration: BoxDecoration(
                      color: containerBg,
                      borderRadius: BorderRadius.circular(16),
                      boxShadow: [
                        BoxShadow(
                          color: isDark ? Colors.black12 : Colors.grey.withOpacity(0.1),
                          blurRadius: 6,
                          offset: const Offset(0, 2),
                        )
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          "User Management",
                          style: theme.textTheme.titleLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: textColor,
                          ),
                        ),
                        const SizedBox(height: 20),

                        TextField(
                          onChanged: (val) {
                            setState(() {
                              _searchText = val;
                            });
                          },
                          decoration: InputDecoration(
                            hintText: 'Search by username...',
                            filled: true,
                            fillColor: cardColor,
                            prefixIcon: Icon(Icons.search, color: accent),
                            border: OutlineInputBorder(
                              borderRadius: BorderRadius.circular(14),
                              borderSide: BorderSide(color: accent.withOpacity(0.2)),
                            ),
                            contentPadding: const EdgeInsets.symmetric(vertical: 0, horizontal: 16),
                          ),
                          style: TextStyle(color: textColor),
                        ),
                        const SizedBox(height: 16),

                        Align(
                          alignment: Alignment.centerRight,
                          child: ElevatedButton.icon(
                            onPressed: () async {
                              final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
                              final bool isDark = themeProvider.themeMode == ThemeMode.dark;
                              final Color dialogBg = isDark ? const Color(0xFF1E293B) : Colors.white;
                              final Color dialogText = isDark ? const Color(0xFFE5E7EB) : Colors.black;
                              final Color accent = const Color(0xFF4F7BFF);
                              TextEditingController newUsernameController = TextEditingController();
                              TextEditingController newPasswordController = TextEditingController();
                              showDialog(
                                context: context,
                                barrierDismissible: false,
                                builder: (context) => AlertDialog(
                                  backgroundColor: dialogBg,
                                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                                  title: Text('Add New User', textAlign: TextAlign.center, style: TextStyle(color: dialogText)),
                                  content: Column(
                                    mainAxisSize: MainAxisSize.min,
                                    children: [
                                      TextField(
                                        controller: newUsernameController,
                                        decoration: InputDecoration(
                                          labelText: 'Username',
                                          border: OutlineInputBorder(),
                                        ),
                                      ),
                                      const SizedBox(height: 12),
                                      TextField(
                                        controller: newPasswordController,
                                        obscureText: true,
                                        decoration: InputDecoration(
                                          labelText: 'Password',
                                          border: OutlineInputBorder(),
                                        ),
                                      ),
                                    ],
                                  ),
                                  actionsAlignment: MainAxisAlignment.center,
                                  actions: [
                                    TextButton(
                                      onPressed: () => Navigator.pop(context),
                                      child: Text('Cancel', style: TextStyle(color: accent)),
                                    ),
                                    ElevatedButton(
                                      style: ElevatedButton.styleFrom(
                                        backgroundColor: accent,
                                        foregroundColor: Colors.white,
                                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                                      ),
                                      onPressed: () async {
                                        if (newUsernameController.text.isNotEmpty && newPasswordController.text.isNotEmpty) {
                                          setState(() {
                                            doctors.add({
                                              'username': newUsernameController.text,
                                              'password': newPasswordController.text,
                                            });
                                          });
                                          await _saveDoctors();
                                          Navigator.pop(context);
                                          ScaffoldMessenger.of(context).showSnackBar(
                                            const SnackBar(content: Text('User added successfully!')),
                                          );
                                        }
                                      },
                                      child: const Text('Add User', style: TextStyle(color: Colors.white)),
                                    ),
                                  ],
                                ),
                              );
                            },
                            icon: const Icon(Icons.person_add),
                            label: const Text("Add New User"),
                            style: ElevatedButton.styleFrom(
                              backgroundColor: accent,
                              foregroundColor: Colors.white,
                              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
                            ),
                          ),
                        ),

                        const SizedBox(height: 20),
                        const Divider(),

                        Builder(
                          builder: (context) {
                            List<Map<String, String>> filteredUsers = _searchText == null || _searchText!.isEmpty
                                ? doctors
                                : doctors.where((user) => user['username']!.toLowerCase().contains(_searchText!.toLowerCase())).toList();
                            int usersPerPage = 10;
                            int pageCount = (filteredUsers.length / usersPerPage).ceil();
                            int currentPage = _currentPage;
                            int start = currentPage * usersPerPage;
                            int end = start + usersPerPage;
                            if (end > filteredUsers.length) end = filteredUsers.length;
                            List<Map<String, String>> pageUsers = filteredUsers.sublist(start, end);

                            return Column(
                              children: [
                                pageUsers.isEmpty
                                    ? const Text("No users found.")
                                    : ListView.separated(
                                        shrinkWrap: true,
                                        physics: const NeverScrollableScrollPhysics(),
                                        itemCount: pageUsers.length,
                                        separatorBuilder: (context, i) => const Divider(),
                                        itemBuilder: (context, index) {
                                          final user = pageUsers[index];
                                          return ListTile(
                                            title: Text(user['username']!),
                                            subtitle: Text(user['password']!),
                                            trailing: Row(
                                              mainAxisSize: MainAxisSize.min,
                                              children: [
                                                IconButton(
                                                  icon: const Icon(Icons.edit, color: Colors.blue),
                                                  onPressed: () {
                                                    final themeProvider = Provider.of<ThemeProvider>(context, listen: false);
                                                    final bool isDark = themeProvider.themeMode == ThemeMode.dark;
                                                    final Color dialogBg = isDark ? const Color(0xFF1E293B) : Colors.white;
                                                    final Color dialogText = isDark ? const Color(0xFFE5E7EB) : Colors.black;
                                                    TextEditingController editUsernameController = TextEditingController(text: user['username']);
                                                    TextEditingController editPasswordController = TextEditingController(text: user['password']);
                                                    showDialog(
                                                      context: context,
                                                      barrierDismissible: false,
                                                      builder: (context) => AlertDialog(
                                                        backgroundColor: dialogBg,
                                                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(24)),
                                                        title: Text('Edit User', textAlign: TextAlign.center, style: TextStyle(color: dialogText)),
                                                        content: Column(
                                                          mainAxisSize: MainAxisSize.min,
                                                          children: [
                                                            TextField(
                                                              controller: editUsernameController,
                                                              decoration: InputDecoration(
                                                                labelText: 'Username',
                                                                border: OutlineInputBorder(),
                                                              ),
                                                            ),
                                                            const SizedBox(height: 12),
                                                            TextField(
                                                              controller: editPasswordController,
                                                              obscureText: true,
                                                              decoration: InputDecoration(
                                                                labelText: 'Password',
                                                                border: OutlineInputBorder(),
                                                              ),
                                                            ),
                                                          ],
                                                        ),
                                                        actionsAlignment: MainAxisAlignment.center,
                                                        actions: [
                                                          TextButton(
                                                            onPressed: () => Navigator.pop(context),
                                                            child: Text('Cancel', style: TextStyle(color: accent)),
                                                          ),
                                                          ElevatedButton(
                                                            style: ElevatedButton.styleFrom(
                                                              backgroundColor: accent,
                                                              foregroundColor: Colors.white,
                                                              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(18)),
                                                            ),
                                                            onPressed: () {
                                                              setState(() {
                                                                doctors[start + index] = {
                                                                  'username': editUsernameController.text,
                                                                  'password': editPasswordController.text,
                                                                };
                                                              });
                                                              Navigator.pop(context);
                                                              ScaffoldMessenger.of(context).showSnackBar(
                                                                const SnackBar(content: Text('User updated successfully!')),
                                                              );
                                                            },
                                                            child: const Text('Save', style: TextStyle(color: Colors.white)),
                                                          ),
                                                        ],
                                                      ),
                                                    );
                                                  },
                                                ),
                                                IconButton(
                                                  icon: const Icon(Icons.delete, color: Colors.red),
                                                  onPressed: () async {
                                                    await deleteDoctor(start + index);
                                                  },
                                                ),
                                              ],
                                            ),
                                          );
                                        },
                                      ),
                                if (pageCount > 1)
                                  Padding(
                                    padding: const EdgeInsets.symmetric(vertical: 12),
                                    child: Row(
                                      mainAxisAlignment: MainAxisAlignment.center,
                                      children: List.generate(pageCount, (i) => Padding(
                                        padding: const EdgeInsets.symmetric(horizontal: 2),
                                        child: ElevatedButton(
                                          style: ElevatedButton.styleFrom(
                                            backgroundColor: i == currentPage ? accent : cardColor,
                                            foregroundColor: Colors.white,
                                            minimumSize: const Size(36, 36),
                                            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                                            elevation: 0,
                                          ),
                                          onPressed: () {
                                            setState(() {
                                              _currentPage = i;
                                            });
                                          },
                                          child: Text("${i + 1}"),
                                        ),
                                      )),
                                    ),
                                  ),
                              ],
                            );
                          },
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
    );
  }
}
