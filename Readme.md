> Simple http server, serve api `/face-validation` to detect and crop face image.

**request**

```bash
curl --location 'http://localhost:8000/face-validation' \
--header 'Content-Type: application/json' \
--data '{
    "image_base64": "<base64-encoded-image>",
    "confidence_threshold": 0.7,
    "face_area_threshold": 0.02
}'
```

**response**

```json
{
  "image": "<base64-encoded-cropped-image>"
}
```

# Build 🐳

```bash
$ docker build -t face-validation .
$ docker run -p 8000:8000 face-validation
```
