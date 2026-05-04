#!/usr/bin/env python3
"""
PDB 分子 3D 结构查看器
- 实时渲染 PDB 分子的 3D 结构
- 鼠标拖拽旋转角度
- 滚轮缩放
- 保存图像
"""
import sys
import os
import math
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QComboBox, QCheckBox, QSlider,
    QStatusBar, QGroupBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QSurfaceFormat, QImage, QPainter
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

# ─── PDB 解析 ───────────────────────────────────────────────────────────────

# CPK 颜色方案 (元素 → RGB)
ELEMENT_COLORS = {
    'C':  (0.6, 0.6, 0.6),   # 灰色
    'N':  (0.2, 0.2, 0.8),   # 蓝色
    'O':  (0.8, 0.1, 0.1),   # 红色
    'S':  (0.9, 0.8, 0.1),   # 黄色
    'H':  (0.9, 0.9, 0.9),   # 白色
    'P':  (0.9, 0.5, 0.0),   # 橙色
    'FE': (0.8, 0.4, 0.0),   # 铁
    'ZN': (0.5, 0.5, 0.8),   # 锌
    'CA': (0.3, 0.8, 0.3),   # 钙
    'MG': (0.3, 0.8, 0.3),   # 镁
    'CL': (0.2, 0.8, 0.2),   # 氯
    'NA': (0.5, 0.5, 0.9),   # 钠
    'K':  (0.5, 0.5, 0.9),   # 钾
    'MN': (0.6, 0.4, 0.2),   # 锰
    'CO': (0.6, 0.4, 0.2),   # 钴
    'NI': (0.6, 0.4, 0.2),   # 镍
    'CU': (0.6, 0.4, 0.2),   # 铜
}

# 残基颜色方案 (按残基类型)
RESIDUE_COLORS = {
    'ALA': (0.8, 0.8, 0.8), 'ARG': (0.2, 0.2, 0.8), 'ASN': (0.2, 0.8, 0.2),
    'ASP': (0.8, 0.2, 0.2), 'CYS': (0.8, 0.8, 0.2), 'GLN': (0.2, 0.8, 0.2),
    'GLU': (0.8, 0.2, 0.2), 'GLY': (0.9, 0.9, 0.9), 'HIS': (0.5, 0.5, 0.8),
    'ILE': (0.5, 0.5, 0.5), 'LEU': (0.5, 0.5, 0.5), 'LYS': (0.2, 0.2, 0.8),
    'MET': (0.8, 0.8, 0.2), 'PHE': (0.5, 0.5, 0.5), 'PRO': (0.8, 0.5, 0.2),
    'SER': (0.2, 0.8, 0.2), 'THR': (0.2, 0.8, 0.2), 'TRP': (0.5, 0.5, 0.5),
    'TYR': (0.5, 0.5, 0.5), 'VAL': (0.5, 0.5, 0.5),
}

# 原子半径 (Å)
ATOM_RADII = {
    'C': 0.30, 'N': 0.28, 'O': 0.27, 'S': 0.35, 'H': 0.15,
    'P': 0.31, 'FE': 0.35, 'ZN': 0.35, 'CA': 0.35, 'MG': 0.35,
}

# 共价键长度阈值 (Å) — 超过此距离不建键
BOND_CUTOFF = 2.0
# 同残基内稍宽松
BOND_CUTOFF_INTRA = 2.5


class Atom:
    __slots__ = ('serial', 'name', 'resname', 'chain', 'resseq',
                 'x', 'y', 'z', 'element', 'bfactor', 'occupancy')
    def __init__(self, serial, name, resname, chain, resseq, x, y, z,
                 element, bfactor=0.0, occupancy=1.0):
        self.serial = serial
        self.name = name.strip()
        self.resname = resname.strip()
        self.chain = chain.strip()
        self.resseq = resseq
        self.x = x
        self.y = y
        self.z = z
        self.element = element.strip().upper()
        self.bfactor = bfactor
        self.occupancy = occupancy


class Bond:
    __slots__ = ('i', 'j', 'order')
    def __init__(self, i, j, order=1):
        self.i = i
        self.j = j
        self.order = order


def parse_pdb(filepath):
    """解析 PDB 文件，返回 atoms 列表"""
    atoms = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    serial = int(line[6:11])
                    name = line[12:16]
                    resname = line[17:20]
                    chain = line[21]
                    resseq = int(line[22:26])
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    occupancy = float(line[54:60]) if len(line) > 54 else 1.0
                    bfactor = float(line[60:66]) if len(line) > 60 else 0.0
                    element = line[76:78].strip() if len(line) > 76 else ''
                    if not element:
                        element = name.strip()[0]
                    atoms.append(Atom(serial, name, resname, chain, resseq,
                                      x, y, z, element, bfactor, occupancy))
                except (ValueError, IndexError):
                    continue
    return atoms


def build_bonds(atoms):
    """基于距离自动构建化学键"""
    bonds = []
    n = len(atoms)
    if n == 0:
        return bonds

    # 按残基分组加速
    residue_groups = {}
    for idx, atom in enumerate(atoms):
        key = (atom.chain, atom.resseq, atom.resname)
        residue_groups.setdefault(key, []).append(idx)

    # 同残基内建键
    for key, indices in residue_groups.items():
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                i, j = indices[ii], indices[jj]
                dx = atoms[i].x - atoms[j].x
                dy = atoms[i].y - atoms[j].y
                dz = atoms[i].z - atoms[j].z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < BOND_CUTOFF_INTRA:
                    bonds.append(Bond(i, j))

    # 不同残基间（肽键等）用较严格阈值
    # 只检查相邻残基
    sorted_residues = sorted(residue_groups.keys(),
                             key=lambda k: (k[0], k[1]))
    for ri in range(len(sorted_residues) - 1):
        k1, k2 = sorted_residues[ri], sorted_residues[ri + 1]
        if k1[0] != k2[0] or abs(k1[1] - k2[1]) > 1:
            continue
        for i in residue_groups[k1]:
            for j in residue_groups[k2]:
                dx = atoms[i].x - atoms[j].x
                dy = atoms[i].y - atoms[j].y
                dz = atoms[i].z - atoms[j].z
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < BOND_CUTOFF:
                    bonds.append(Bond(i, j))

    return bonds


def get_atom_color(atom, color_mode):
    """根据颜色模式获取原子颜色"""
    if color_mode == 'element':
        return ELEMENT_COLORS.get(atom.element, (0.5, 0.5, 0.5))
    elif color_mode == 'residue':
        return RESIDUE_COLORS.get(atom.resname, (0.5, 0.5, 0.5))
    elif color_mode == 'chain':
        chain_colors = {
            'A': (0.2, 0.4, 0.8), 'B': (0.8, 0.2, 0.2),
            'C': (0.2, 0.8, 0.2), 'D': (0.8, 0.8, 0.2),
            'E': (0.8, 0.2, 0.8), 'F': (0.2, 0.8, 0.8),
        }
        return chain_colors.get(atom.chain, (0.5, 0.5, 0.5))
    elif color_mode == 'bfactor':
        # B-factor 热力图: 蓝(低) → 红(高)
        t = min(1.0, max(0.0, atom.bfactor / 100.0))
        return (t, 0.2, 1.0 - t)
    return (0.5, 0.5, 0.5)


# ─── OpenGL 分子渲染 Widget ─────────────────────────────────────────────────

class MolGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.atoms = []
        self.bonds = []
        self.center = np.array([0.0, 0.0, 0.0])
        self.scale = 1.0
        self.color_mode = 'element'
        self.render_mode = 'ball_and_stick'  # ball_and_stick / sphere / wireframe
        self.show_hydrogen = False
        self.show_labels = False

        # 旋转矩阵 (trackball)
        self.rotation = np.eye(4, dtype=np.float32)
        self.last_pos = None

        # 平移
        self.translation = np.array([0.0, 0.0, 0.0])

        # 光照参数
        self.ambient = [0.3, 0.3, 0.3, 1.0]
        self.diffuse = [0.7, 0.7, 0.7, 1.0]
        self.specular = [0.4, 0.4, 0.4, 1.0]
        self.shininess = 50.0

        # 显示列表
        self.display_list = None
        self.needs_rebuild = True

        self.setMinimumSize(400, 300)

    def load_molecule(self, filepath):
        """加载 PDB 文件"""
        self.atoms = parse_pdb(filepath)
        self.bonds = build_bonds(self.atoms)
        if self.atoms:
            coords = np.array([[a.x, a.y, a.z] for a in self.atoms])
            self.center = coords.mean(axis=0)
            # 计算合适的缩放
            span = coords.max(axis=0) - coords.min(axis=0)
            max_span = max(span)
            self.scale = 10.0 / max_span if max_span > 0 else 1.0
        self.rotation = np.eye(4, dtype=np.float32)
        self.translation = np.array([0.0, 0.0, 0.0])
        self.needs_rebuild = True
        self.update()

    def initializeGL(self):
        glClearColor(0.1, 0.1, 0.15, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_NORMALIZE)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, [5.0, 5.0, 10.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, self.ambient)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, self.diffuse)
        glLightfv(GL_LIGHT0, GL_SPECULAR, self.specular)
        glMaterialf(GL_FRONT, GL_SHININESS, self.shininess)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        aspect = w / h if h > 0 else 1.0
        gluPerspective(45.0, aspect, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 相机位置
        gluLookAt(0, 0, 30, 0, 0, 0, 0, 1, 0)

        # 应用旋转
        glMultMatrixf(self.rotation)

        # 应用平移
        glTranslatef(self.translation[0], self.translation[1], self.translation[2])

        # 应用缩放和居中
        glScalef(self.scale, self.scale, self.scale)
        glTranslatef(-self.center[0], -self.center[1], -self.center[2])

        # 重新构建显示列表
        if self.needs_rebuild or self.display_list is None:
            self._build_display_list()
            self.needs_rebuild = False

        if self.display_list is not None:
            glCallList(self.display_list)

    def _build_display_list(self):
        """构建 OpenGL 显示列表"""
        if self.display_list is not None:
            glDeleteLists(self.display_list, 1)
        self.display_list = glGenLists(1)
        glNewList(self.display_list, GL_COMPILE)

        # 绘制键
        if self.render_mode != 'sphere':
            glDisable(GL_LIGHTING)
            glBegin(GL_LINES)
            for bond in self.bonds:
                a1, a2 = self.atoms[bond.i], self.atoms[bond.j]
                if not self.show_hydrogen and (a1.element == 'H' or a2.element == 'H'):
                    continue
                c1 = get_atom_color(a1, self.color_mode)
                c2 = get_atom_color(a2, self.color_mode)
                glColor3f(*c1)
                glVertex3f(a1.x, a1.y, a1.z)
                glColor3f(*c2)
                glVertex3f(a2.x, a2.y, a2.z)
            glEnd()
            glEnable(GL_LIGHTING)

        # 绘制原子
        quad = gluNewQuadric()
        gluQuadricNormals(quad, GLU_SMOOTH)

        for atom in self.atoms:
            if not self.show_hydrogen and atom.element == 'H':
                continue

            color = get_atom_color(atom, self.color_mode)
            glColor3f(*color)

            if self.render_mode == 'ball_and_stick':
                radius = ATOM_RADII.get(atom.element, 0.25) * 0.4
            elif self.render_mode == 'sphere':
                radius = ATOM_RADII.get(atom.element, 0.25) * 1.0
            else:  # wireframe
                radius = ATOM_RADII.get(atom.element, 0.25) * 0.2

            glPushMatrix()
            glTranslatef(atom.x, atom.y, atom.z)
            gluSphere(quad, radius, 12, 12)
            glPopMatrix()

        gluDeleteQuadric(quad)
        glEndList()

    # ─── 鼠标交互 (trackball 旋转) ─────────────────────────

    def mousePressEvent(self, event):
        self.last_pos = event.position()

    def mouseMoveEvent(self, event):
        if self.last_pos is None:
            return

        dx = event.position().x() - self.last_pos.x()
        dy = event.position().y() - self.last_pos.y()
        self.last_pos = event.position()

        if event.buttons() & Qt.MouseButton.LeftButton:
            # 旋转
            self._trackball_rotate(dx, dy)
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            # 平移
            self.translation[0] += dx * 0.05
            self.translation[1] -= dy * 0.05
        elif event.buttons() & Qt.MouseButton.RightButton:
            # 平移
            self.translation[0] += dx * 0.05
            self.translation[1] -= dy * 0.05

        self.update()

    def mouseReleaseEvent(self, event):
        self.last_pos = None

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scale *= 1.1
        else:
            self.scale /= 1.1
        self.update()

    def _trackball_rotate(self, dx, dy):
        """Trackball 旋转实现"""
        angle_x = dy * 0.5
        angle_y = dx * 0.5

        # 构造旋转矩阵
        cx, sx = math.cos(math.radians(angle_x)), math.sin(math.radians(angle_x))
        cy, sy = math.cos(math.radians(angle_y)), math.sin(math.radians(angle_y))

        rx = np.array([
            [1, 0, 0, 0],
            [0, cx, -sx, 0],
            [0, sx, cx, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        ry = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        self.rotation = ry @ rx @ self.rotation

    def save_image(self, filepath):
        """保存当前渲染结果为图像"""
        image = self.grabFramebuffer()
        image.save(filepath)
        return filepath


# ─── 主窗口 ──────────────────────────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI4S 分子 3D 结构查看器")
        self.setMinimumSize(900, 700)

        # 中央 Widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 左侧: OpenGL 渲染区
        self.gl_widget = MolGLWidget()
        main_layout.addWidget(self.gl_widget, stretch=1)

        # 右侧: 控制面板
        ctrl_panel = QWidget()
        ctrl_panel.setMaximumWidth(220)
        ctrl_layout = QVBoxLayout(ctrl_panel)

        # ─ 分子信息
        info_group = QGroupBox("分子信息")
        info_layout = QVBoxLayout(info_group)
        self.lbl_atoms = QLabel("原子数: —")
        self.lbl_bonds = QLabel("化学键数: —")
        self.lbl_residues = QLabel("残基数: —")
        self.lbl_chains = QLabel("链数: —")
        info_layout.addWidget(self.lbl_atoms)
        info_layout.addWidget(self.lbl_bonds)
        info_layout.addWidget(self.lbl_residues)
        info_layout.addWidget(self.lbl_chains)
        ctrl_layout.addWidget(info_group)

        # ─ 颜色模式
        color_group = QGroupBox("着色模式")
        color_layout = QVBoxLayout(color_group)
        self.combo_color = QComboBox()
        self.combo_color.addItems(["元素(CPK)", "残基", "链", "B-factor"])
        self.combo_color.currentIndexChanged.connect(self._on_color_changed)
        color_layout.addWidget(self.combo_color)
        ctrl_layout.addWidget(color_group)

        # ─ 渲染模式
        render_group = QGroupBox("渲染模式")
        render_layout = QVBoxLayout(render_group)
        self.combo_render = QComboBox()
        self.combo_render.addItems(["球棍模型", "空间填充", "线框模型"])
        self.combo_render.currentIndexChanged.connect(self._on_render_changed)
        render_layout.addWidget(self.combo_render)
        ctrl_layout.addWidget(render_group)

        # ─ 显示选项
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout(display_group)
        self.chk_hydrogen = QCheckBox("显示氢原子")
        self.chk_hydrogen.toggled.connect(self._on_hydrogen_toggled)
        display_layout.addWidget(self.chk_hydrogen)
        ctrl_layout.addWidget(display_group)

        # ─ 操作按钮
        btn_group = QGroupBox("操作")
        btn_layout = QVBoxLayout(btn_group)

        btn_open = QPushButton("📂 打开 PDB 文件")
        btn_open.clicked.connect(self._open_file)
        btn_layout.addWidget(btn_open)

        btn_save = QPushButton("💾 保存图像")
        btn_save.clicked.connect(self._save_image)
        btn_layout.addWidget(btn_save)

        btn_reset = QPushButton("🔄 重置视角")
        btn_reset.clicked.connect(self._reset_view)
        btn_layout.addWidget(btn_reset)

        btn_bg = QPushButton("🎨 切换背景色")
        btn_bg.clicked.connect(self._toggle_bg)
        btn_layout.addWidget(btn_bg)

        ctrl_layout.addWidget(btn_group)
        ctrl_layout.addStretch()

        # 操作提示
        hint = QLabel("🖱 左键: 旋转\n🖱 中键/右键: 平移\n🖱 滚轮: 缩放")
        hint.setStyleSheet("color: #888; font-size: 11px;")
        ctrl_layout.addWidget(hint)

        main_layout.addWidget(ctrl_panel)

        # 状态栏
        self.statusBar().showMessage("就绪 — 拖放 PDB 文件或点击「打开」")

        # 深色主题
        self.setStyleSheet("""
            QMainWindow, QWidget { background: #1e1e2e; color: #cdd6f4; }
            QGroupBox { border: 1px solid #45475a; border-radius: 6px;
                        margin-top: 8px; padding-top: 14px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { background: #313244; border: 1px solid #45475a;
                          border-radius: 4px; padding: 6px 12px; }
            QPushButton:hover { background: #45475a; }
            QComboBox { background: #313244; border: 1px solid #45475a;
                        border-radius: 4px; padding: 4px; }
            QCheckBox { spacing: 6px; }
            QLabel { font-size: 12px; }
        """)

        # 自动加载默认 PDB
        default_pdb = os.path.join(os.path.dirname(__file__), 'target.pdb')
        if os.path.exists(default_pdb):
            self._load_pdb(default_pdb)

    def _load_pdb(self, filepath):
        """加载 PDB 文件并更新信息"""
        self.gl_widget.load_molecule(filepath)
        atoms = self.gl_widget.atoms
        bonds = self.gl_widget.bonds

        residues = set((a.chain, a.resseq) for a in atoms)
        chains = set(a.chain for a in atoms)

        self.lbl_atoms.setText(f"原子数: {len(atoms)}")
        self.lbl_bonds.setText(f"化学键数: {len(bonds)}")
        self.lbl_residues.setText(f"残基数: {len(residues)}")
        self.lbl_chains.setText(f"链数: {len(chains)}")

        self.statusBar().showMessage(
            f"已加载: {os.path.basename(filepath)} | "
            f"{len(atoms)} 原子, {len(bonds)} 键, {len(residues)} 残基"
        )
        self.setWindowTitle(f"AI4S 分子查看器 — {os.path.basename(filepath)}")

    def _open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "打开 PDB 文件", "", "PDB 文件 (*.pdb *.pdbqt *.ent);;所有文件 (*)"
        )
        if filepath:
            self._load_pdb(filepath)

    def _save_image(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "保存图像", "molecule.png",
            "PNG 图像 (*.png);;JPEG 图像 (*.jpg);;所有文件 (*)"
        )
        if filepath:
            self.gl_widget.save_image(filepath)
            self.statusBar().showMessage(f"图像已保存: {filepath}")

    def _reset_view(self):
        self.gl_widget.rotation = np.eye(4, dtype=np.float32)
        self.gl_widget.translation = np.array([0.0, 0.0, 0.0])
        if self.gl_widget.atoms:
            coords = np.array([[a.x, a.y, a.z] for a in self.gl_widget.atoms])
            span = coords.max(axis=0) - coords.min(axis=0)
            max_span = max(span)
            self.gl_widget.scale = 10.0 / max_span if max_span > 0 else 1.0
        self.gl_widget.update()
        self.statusBar().showMessage("视角已重置")

    def _toggle_bg(self):
        gl = self.gl_widget
        gl.makeCurrent()
        col = glClearColor
        # 读取当前背景色
        current = glGetFloatv(GL_COLOR_CLEAR_VALUE)
        if current[0] < 0.2:  # 深色 → 浅色
            glClearColor(0.9, 0.9, 0.92, 1.0)
        else:  # 浅色 → 深色
            glClearColor(0.1, 0.1, 0.15, 1.0)
        gl.update()

    def _on_color_changed(self, idx):
        modes = ['element', 'residue', 'chain', 'bfactor']
        self.gl_widget.color_mode = modes[idx]
        self.gl_widget.needs_rebuild = True
        self.gl_widget.update()

    def _on_render_changed(self, idx):
        modes = ['ball_and_stick', 'sphere', 'wireframe']
        self.gl_widget.render_mode = modes[idx]
        self.gl_widget.needs_rebuild = True
        self.gl_widget.update()

    def _on_hydrogen_toggled(self, checked):
        self.gl_widget.show_hydrogen = checked
        self.gl_widget.needs_rebuild = True
        self.gl_widget.update()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("AI4S 分子 3D 查看器")

    # 设置 OpenGL 格式
    fmt = QSurfaceFormat()
    fmt.setDepthBufferSize(24)
    fmt.setSamples(4)  # MSAA
    fmt.setVersion(2, 1)
    QSurfaceFormat.setDefaultFormat(fmt)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
