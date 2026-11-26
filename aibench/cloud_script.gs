/**
 * AI Bench – cầu nối lưu kết quả benchmark lên Google Drive.
 *
 * Cài một lần:
 *   1. Mở https://script.google.com → Dự án mới → xóa hết nội dung có sẵn, dán toàn bộ file này vào, bấm Lưu.
 *   2. Triển khai → Tùy chọn triển khai mới → loại "Ứng dụng web".
 *        Thực thi với tư cách: Tôi.   Người có quyền truy cập: Bất kỳ ai.
 *      Bấm Triển khai, cho phép quyền truy cập Google Drive khi được hỏi.
 *   3. Chép "URL ứng dụng web" (…/exec) dán vào ô "Link Apps Script" trong AI Bench.
 *
 * File được lưu trong thư mục FOLDER_ID dưới tài khoản Google của người triển khai.
 * Ai có link /exec mà không có KEY thì không đọc / ghi được. Đừng chia sẻ file này ra ngoài.
 */
const FOLDER_ID = '__FOLDER_ID__';
const KEY = '__KEY__';

function doPost(e) {
  let req;
  try {
    req = JSON.parse(e.postData.contents);
  } catch (err) {
    return out_({ok: false, error: 'Yêu cầu không hợp lệ'});
  }
  if (req.key !== KEY) return out_({ok: false, error: 'Sai khóa: script trên Google không khớp với AI Bench'});
  const folder = DriveApp.getFolderById(FOLDER_ID);

  if (req.action === 'ping') return out_({ok: true, folder: folder.getName()});

  if (req.action === 'list') {
    const files = [];
    const it = folder.getFiles();
    while (it.hasNext()) {
      const f = it.next();
      if (/\.json$/i.test(f.getName())) files.push({name: f.getName(), size: f.getSize(), updated: f.getLastUpdated().toISOString()});
    }
    return out_({ok: true, files: files});
  }

  const name = String(req.name || '');
  if (!/^[\w\-]+\.json$/.test(name)) return out_({ok: false, error: 'Tên file không hợp lệ'});

  if (req.action === 'get') {
    const it = folder.getFilesByName(name);
    if (!it.hasNext()) return out_({ok: false, error: 'Không có file ' + name});
    return out_({ok: true, name: name, content: it.next().getBlob().getDataAsString('UTF-8')});
  }

  if (req.action === 'put') {
    const old = folder.getFilesByName(name);  // ghi đè: bỏ bản cũ vào thùng rác
    while (old.hasNext()) old.next().setTrashed(true);
    const f = folder.createFile(Utilities.newBlob(String(req.content || ''), 'application/json', name));
    return out_({ok: true, name: name, size: f.getSize()});
  }

  return out_({ok: false, error: 'Không rõ yêu cầu ' + req.action});
}

function doGet() {
  return out_({ok: true, app: 'AI Bench cloud'});
}

function out_(o) {
  return ContentService.createTextOutput(JSON.stringify(o)).setMimeType(ContentService.MimeType.JSON);
}
