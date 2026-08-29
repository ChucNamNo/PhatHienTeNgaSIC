import sys
import json
import base64
from pathlib import Path

import numpy as np
import cv2

# -----------------------------------------------------------------------------
# 1. THIẾT LẬP ĐƯỜNG DẪN GỐC & IMPORT MODEL SERVICE
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import Singleton Service từ thư mục services
try:
    from services.model_service import get_model_service, FallDetectionService
except ImportError as exc:
    raise ImportError(
        "Không tìm thấy thư mục 'services'. Đảm bảo tệp model_service.py nằm trong thư mục services/"
    ) from exc

# Khởi tạo instance AI Service và chuẩn hóa đường dẫn thư mục model
ai_service: FallDetectionService = get_model_service(
    strict_mode=False, 
    pose_filename="yolov8n-pose.pt"
)
ai_service.model_dir = ROOT / "ai_models"

# Tự động tìm kiếm file trọng số phù hợp
pose_candidates = [
    ai_service.model_dir / "yolov8n-pose.pt",
    ai_service.model_dir / "yolo26n-pose.pt",
    ai_service.model_dir / "yolo26s-pose.pt"
]
ai_service.pose_path = next((p for p in pose_candidates if p.exists()), ai_service.model_dir / "yolov8n-pose.pt")

config_candidates = [
    ai_service.model_dir / "Best_BiGRU_Attention_Config.npy",
    ai_service.model_dir / "Best BigRU Attention Config.npy"
]
ai_service.config_path = next((p for p in config_candidates if p.exists()), ai_service.model_dir / "Best_BiGRU_Attention_Config.npy")

weights_candidates = [
    ai_service.model_dir / "Best_BiGRU_Attention_Model.pth",
    ai_service.model_dir / "Best BigRU Attention Model.pth"
]
ai_service.classifier_path = next((p for p in weights_candidates if p.exists()), ai_service.model_dir / "Best_BiGRU_Attention_Model.pth")


# -----------------------------------------------------------------------------
# 2. CẤU HÌNH DJANGO SERVER & GUNICORN APPLICATION
# -----------------------------------------------------------------------------
from django.conf import settings
from django.urls import path, re_path
from django.http import JsonResponse, HttpResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.static import serve
from django.core.management import execute_from_command_line
from django.core.wsgi import get_wsgi_application

if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="sic-ai-fallguard-standalone-secret-key",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["*"],
        INSTALLED_APPS=[
            "django.contrib.staticfiles",
            "django.contrib.contenttypes",
            "django.contrib.auth",
        ],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            "django.middleware.clickjacking.XFrameOptionsMiddleware",
        ],
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "DIRS": [ROOT / "templates"],
                "APP_DIRS": True,
                "OPTIONS": {
                    "context_processors": [
                        "django.template.context_processors.static",
                    ],
                },
            },
        ],
        STATIC_URL="/static/",
        STATICFILES_DIRS=[ROOT / "static"],
    )
    import django
    django.setup()

# Khai báo biến application phục vụ Gunicorn trên Render
application = get_wsgi_application()


# -----------------------------------------------------------------------------
# 3. VIEWS VÀ BỘ ĐIỀU HƯỚNG REQUEST
# -----------------------------------------------------------------------------
def index_view(request):
    """Render giao diện dashboard từ templates/index.html."""
    template_path = ROOT / "templates" / "index.html"
    if template_path.exists():
        return render(request, "index.html")
    return HttpResponse("<h3>Không tìm thấy tệp index.html trong thư mục templates/</h3>", status=404)


@csrf_exempt
def predict_view(request):
    """Tiếp nhận frame JPEG nhị phân từ camera.js và truyền qua pipeline AI."""
    if request.method != "POST":
        return JsonResponse({"ok": False, "error": "Chỉ chấp nhận phương thức POST"}, status=405)

    try:
        frame_bgr = None
        session_id = "default_session"

        # 1. Đọc tệp nhị phân gửi qua FormData
        if request.FILES.get("image"):
            image_file = request.FILES.get("image")
            session_id = request.POST.get("session_id", "default_session")
            image_bytes = image_file.read()
            image_np = np.frombuffer(image_bytes, dtype=np.uint8)
            frame_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # 2. Hỗ trợ dự phòng chuỗi Base64
        elif request.body:
            try:
                data = json.loads(request.body.decode("utf-8"))
                session_id = data.get("session_id", "default_session")
                image_data = data.get("image", "")
                if image_data:
                    if "," in image_data:
                        image_data = image_data.split(",", 1)[1]
                    image_bytes = base64.b64decode(image_data)
                    image_np = np.frombuffer(image_bytes, dtype=np.uint8)
                    frame_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
            except Exception:
                pass

        if frame_bgr is None:
            return JsonResponse({"ok": False, "error": "Không thể giải mã hình ảnh từ frame gửi lên."}, status=400)

        # Kiểm tra trạng thái sẵn sàng của AI Service trước khi predict
        if not ai_service.model_loaded:
            return JsonResponse({"ok": False, "error": f"Mô hình AI chưa sẵn sàng: {ai_service.load_error}"}, status=500)

        # Chạy suy luận qua FallDetectionService
        result = ai_service.predict(frame_bgr, session_id=session_id)
        return JsonResponse(result)

    except Exception as e:
        import traceback
        traceback.print_exc()
        # Đảm bảo trả về JSON thay vì HTML 500 error page của Django
        return JsonResponse({"ok": False, "error": f"Lỗi xử lý server: {str(e)}"}, status=500)


@csrf_exempt
def reset_view(request):
    """Đặt lại dữ liệu theo dõi và bộ đệm của phiên hiện tại."""
    if request.method == "POST":
        session_id = request.POST.get("session_id")
        if not session_id and request.body:
            try:
                data = json.loads(request.body.decode("utf-8"))
                session_id = data.get("session_id")
            except Exception:
                pass

        session_id = session_id or "default_session"
        ai_service.reset(session_id)
        return JsonResponse({"ok": True, "status": "reset_success"})
    return JsonResponse({"ok": False, "error": "Method not allowed"}, status=405)


def status_view(request):
    """Cung cấp trạng thái hoạt động của mô hình và phần cứng thực thi."""
    try:
        health_info = ai_service.health(load=True)
        return JsonResponse({
            "ok": health_info.get("ok", True),
            "status": "ready" if health_info.get("model_loaded") else "loading",
            "model_loaded": health_info.get("model_loaded", False),
            "device": health_info.get("device", "CPU"),
            "gpu_name": health_info.get("gpu_name"),
            "threshold": ai_service.threshold * 100
        })
    except Exception as e:
        return JsonResponse({"ok": False, "model_loaded": False, "error": str(e)}, status=500)


# -----------------------------------------------------------------------------
# 4. DANH SÁCH URLS VÀ STATIC FILES ROUTING
# -----------------------------------------------------------------------------
urlpatterns = [
    path("", index_view, name="index"),
    path("predict/", predict_view, name="predict"),
    path("api/predict/", predict_view, name="api_predict"),
    path("detector/predict/", predict_view),
    path("reset/", reset_view, name="reset"),
    path("api/reset/", reset_view, name="api_reset"),
    path("status/", status_view, name="status"),
    path("api/health/", status_view, name="api_health"),
]

# Tự động map thư mục static phục vụ CSS/JS
static_folder = ROOT / "static"
if static_folder.exists():
    urlpatterns.append(
        re_path(r"^static/(?P<path>.*)$", serve, {"document_root": str(static_folder)})
    )


# -----------------------------------------------------------------------------
# 5. KHỞI CHẠY SERVER
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    host = "127.0.0.1"
    port = "8000"

    print("\n" + "=" * 60)
    print("  FALLGUARD AI - DJANGO SERVER ĐANG KHỞI CHẠY")
    print(f"  Truy cập Dashboard: http://{host}:{port}")
    print("=" * 60 + "\n")

    if len(sys.argv) == 1:
        sys.argv = ["app.py", "runserver", f"{host}:{port}", "--noreload"]

    execute_from_command_line(sys.argv)
