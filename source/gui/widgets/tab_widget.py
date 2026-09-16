"""Tab widget that supports detaching tabs into separate windows."""

from PySide6 import QtCore, QtWidgets

from source.log import get_logger

logger = get_logger()


class DetachableTabWidget(QtWidgets.QTabWidget):
    """Custom tab widget that allows detaching tabs into separate windows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.tabBar = self.tabBar()
        self.tabBar.setContextMenuPolicy(QtCore.Qt.ContextMenuPolicy.CustomContextMenu)
        self.tabBar.customContextMenuRequested.connect(self.showContextMenu)
        self.detachedTabs = {}

    def showContextMenu(self, pos):
        index = self.tabBar.tabAt(pos)
        if index >= 0:
            menu = QtWidgets.QMenu()
            detachAction = menu.addAction("Detach")
            detachAction.triggered.connect(lambda: self.detach_tab(index))
            menu.exec(self.tabBar.mapToGlobal(pos))

    def detach_tab(self, index):
        tab_text = self.tabText(index)
        tab_widget = self.widget(index)
        self.removeTab(index)

        # Lazy import to avoid the tab_window/tab_widget circular load.
        from .tab_window import DetachableTabWindow
        detached_window = DetachableTabWindow(tab_widget, tab_text, self)
        detached_window.show()
        self.detachedTabs[tab_text] = detached_window

    def attach_tab(self, tab_text, tab_widget):
        """Attach a previously detached tab back in sorted order by setup number."""
        if tab_text in self.detachedTabs:
            del self.detachedTabs[tab_text]

        # Extract setup number from tab name (e.g. "Setup 1" -> 1)
        import re
        match = re.search(r'(\d+)', tab_text)
        reattach_num = int(match.group(1)) if match else float('inf')

        # Find correct insertion index by comparing setup numbers
        insert_index = self.count()
        for i in range(self.count()):
            existing_text = self.tabText(i)
            existing_match = re.search(r'(\d+)', existing_text)
            if existing_match:
                existing_num = int(existing_match.group(1))
                if reattach_num < existing_num:
                    insert_index = i
                    break

        self.insertTab(insert_index, tab_widget, tab_text)
        self.setCurrentWidget(tab_widget)


