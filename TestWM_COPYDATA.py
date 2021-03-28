
import win32con, win32api, win32gui, ctypes, ctypes.wintypes
class COPYDATASTRUCT(ctypes.Structure):
	_fields_ = [
		('dwData', ctypes.wintypes.LPARAM),
		('cbData', ctypes.wintypes.DWORD),
		('lpData', ctypes.c_void_p)
	]
PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)
class Listener:
	def __init__(self):
		message_map = {
			win32con.WM_COPYDATA: self.OnCopyData
		}
		wc = win32gui.WNDCLASS()
		wc.lpfnWndProc = message_map
		wc.lpszClassName = 'MyWindowClass'
		hinst = wc.hInstance = win32api.GetModuleHandle(None)
		classAtom = win32gui.RegisterClass(wc)
		self.hwnd = win32gui.CreateWindow (
			classAtom,
			"win32gui test",
			0,
			0, 
			0,
			win32con.CW_USEDEFAULT, 
			win32con.CW_USEDEFAULT,
			0, 
			0,
			hinst, 
			None
		)
		print(self.hwnd)
	def OnCopyData(self, hwnd, msg, wparam, lparam):
		print(hwnd)
		print(msg)
		print(wparam)
		print(lparam)
		pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)
		print(pCDS.contents.dwData)
		print(pCDS.contents.cbData)
		print(ctypes.wstring_at(pCDS.contents.lpData))
		return 1
l = Listener()
win32gui.PumpMessages()
