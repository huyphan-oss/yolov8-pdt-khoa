# Báo cáo các phương pháp optimize trong `yolov8n_custom_dsc_p45`

File model: `cfg/models/v8/yolov8n_custom_dsc_p45.yaml`

File đối chiếu: `cfg/models/v8/yolov8.yaml` với scale `n`.

Báo cáo này tập trung vào các phương pháp tối ưu hóa được sử dụng trong model custom và hiệu quả cụ thể ở từng lớp. Số liệu được tính bằng cách dựng model trong repo hiện tại, so sánh YOLOv8n gốc và model custom với cùng `nc=1` để công bằng.

## Tổng quan hiệu quả

| Model | Số tham số |
| --- | ---: |
| YOLOv8n gốc, `nc=1` | `3,011,043` |
| `yolov8n_custom_dsc_p45`, `nc=1` | `613,427` |
| Giảm | `2,397,616` |
| Tỉ lệ giảm | `79.63%` |

Như vậy bản custom giảm khoảng `79.63%` tham số so với YOLOv8n gốc khi cùng cấu hình `nc=1`.

Điểm cần lưu ý: bản mới không khai báo `reg_max`, nên detect head dùng mặc định `reg_max=16`. Vì vậy phần tối ưu không đến từ việc giảm `reg_max`, mà chủ yếu đến từ thay Conv/C2f/Detect bằng các module nhẹ hơn.

## Các phương pháp optimize được sử dụng

### 1. `DSC_LR_Conv`

`DSC_LR_Conv` thay một Conv 3x3 thường bằng ba bước nhẹ hơn:

```text
Depthwise 3x3
-> Low-rank reduce 1x1
-> Low-rank expand 1x1
```

Conv 3x3 thường từ `c1` sang `c2` dùng:

```text
3 * 3 * c1 * c2
```

Nó rất nặng vì mỗi output channel nhìn toàn bộ input channel bằng kernel 3x3.

`DSC_LR_Conv` làm nhẹ bằng cách:

- dùng depthwise 3x3 để xử lý không gian riêng từng kênh;
- nén kênh xuống `rank`;
- mở rộng từ `rank` lên số kênh đầu ra.

Trong YAML này, `DSC_LR_Conv` dùng:

```text
rank_ratio = 0.125
rank_min = 8
```

Rank được tính:

```text
rank = max(8, int(min(c1, c2) * 0.125))
```

### 2. `LRFuseC2f`

`LRFuseC2f` là biến thể nhẹ của `C2f`.

Khối `C2f` gốc có các phần chính:

```text
cv1 -> bottleneck(s) -> concat -> cv2
```

`LRFuseC2f` giữ gần giống `C2f`, nhưng thay lớp fusion cuối `cv2` bằng low-rank 1x1 Conv.

Ý nghĩa: vẫn giữ cấu trúc xử lý đặc trưng của C2f, nhưng làm nhẹ bước trộn đặc trưng cuối sau concat.

### 3. `LRTuckerC2f`

`LRTuckerC2f` tối ưu mạnh hơn `LRFuseC2f`.

Nó dùng:

- low-rank 1x1 Conv ở đầu;
- `TuckerBottleneck` bên trong;
- low-rank 1x1 Conv ở cuối.

Trong `TuckerBottleneck`, Conv 3x3 được xấp xỉ bằng:

```text
1x1 reduce -> 3x1 -> 1x3 -> 1x1 expand
```

Bản mới pull về còn giảm expansion `e` của hai khối P5 từ `0.5` xuống `0.125`, nên số kênh ẩn trong P5 nhỏ hơn nhiều.

Ví dụ dễ hiểu:

```text
Nếu output danh nghĩa là 1024:
e = 0.5   -> kênh ẩn khoảng 512
e = 0.125 -> kênh ẩn khoảng 128
```

Vì P5 là stage có số kênh cao nhất, giảm `e` ở P5 giúp giảm tham số rất mạnh.

### 4. `DSCDetect`

YOLOv8n gốc dùng `Detect`.

Trong `Detect`, mỗi scale P3/P4/P5 có hai nhánh:

```text
nhánh bbox
nhánh class
```

`DSCDetect` vẫn giữ hai nhánh đó, vẫn giữ cách decode bbox của YOLOv8, nhưng thay các Conv 3x3 trong hai nhánh bằng `DSC_LR_Conv`.

Với model hiện tại:

```text
nc = 1
reg_max = 16
c2 = 64 cho nhánh bbox
c3 = 64 cho nhánh class
```

Mỗi scale detect có dạng:

```text
Nhánh bbox:
DSC_LR_Conv(x, 64)
-> DSC_LR_Conv(64, 64)
-> Conv2d(64, 64)

Nhánh class:
DSC_LR_Conv(x, 64)
-> DSC_LR_Conv(64, 64)
-> Conv2d(64, 1)
```

Trong đó `x` lần lượt là:

```text
P3: 64 kênh
P4: 128 kênh
P5: 256 kênh
```

## Hiệu quả cụ thể từng lớp

### Bảng tổng hợp các lớp được thay thế

| Layer | YOLOv8n gốc | Custom | Params gốc | Params custom | Giảm | Tỉ lệ giảm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 5 | `Conv` | `DSC_LR_Conv` | `73,984` | `2,512` | `71,472` | `96.60%` |
| 6 | `C2f` | `LRFuseC2f` | `197,632` | `62,208` | `135,424` | `68.52%` |
| 7 | `Conv` | `DSC_LR_Conv` | `295,424` | `8,096` | `287,328` | `97.26%` |
| 8 | `C2f` | `LRTuckerC2f` | `460,288` | `9,064` | `451,224` | `98.03%` |
| 12 | `C2f` | `LRFuseC2f` | `148,224` | `144,256` | `3,968` | `2.68%` |
| 16 | `Conv` | `DSC_LR_Conv` | `36,992` | `1,872` | `35,120` | `94.94%` |
| 18 | `C2f` | `LRFuseC2f` | `123,648` | `42,080` | `81,568` | `65.97%` |
| 19 | `Conv` | `DSC_LR_Conv` | `147,712` | `5,792` | `141,920` | `96.08%` |
| 21 | `C2f` | `LRTuckerC2f` | `493,056` | `10,088` | `482,968` | `97.95%` |
| 22 | `Detect` | `DSCDetect` | `751,507` | `44,883` | `706,624` | `94.03%` |

Tổng riêng các lớp được thay thế:

| Nhóm lớp | Params gốc | Params custom | Giảm | Tỉ lệ giảm |
| --- | ---: | ---: | ---: | ---: |
| Các lớp thay thế | `2,728,467` | `330,851` | `2,397,616` | `87.87%` |

Vì các layer còn lại gần như giữ nguyên, mức giảm ở nhóm lớp thay thế cũng chính là phần tạo ra mức giảm tổng thể của model.

## Hiệu quả theo từng phương pháp

| Phương pháp | Layer áp dụng | Params gốc | Params custom | Giảm | Tỉ lệ giảm |
| --- | --- | ---: | ---: | ---: | ---: |
| `DSC_LR_Conv` | 5, 7, 16, 19 | `554,112` | `18,272` | `535,840` | `96.70%` |
| `LRFuseC2f` | 6, 12, 18 | `469,504` | `248,544` | `220,960` | `47.06%` |
| `LRTuckerC2f` | 8, 21 | `953,344` | `19,152` | `934,192` | `97.99%` |
| `DSCDetect` | 22 | `751,507` | `44,883` | `706,624` | `94.03%` |

Nhìn vào bảng này có thể thấy:

- `DSC_LR_Conv` giảm rất mạnh ở các lớp Conv downsample.
- `LRTuckerC2f` giảm mạnh nhất ở các khối P5 vì P5 có số kênh cao và bản mới dùng `e=0.125`.
- `DSCDetect` giảm rất mạnh detect head.
- `LRFuseC2f` giảm vừa phải hơn, vì nó chỉ thay phần fusion cuối thay vì thay toàn bộ C2f.

## Phân tích cụ thể `DSC_LR_Conv`

Các lớp `DSC_LR_Conv` trong YAML:

```yaml
5:  [-1, 1, DSC_LR_Conv, [512, 3, 2, True, null, 0.125, 8]]
7:  [-1, 1, DSC_LR_Conv, [1024, 3, 2, True, null, 0.125, 8]]
16: [-1, 1, DSC_LR_Conv, [256, 3, 2, True, null, 0.125, 8]]
19: [-1, 1, DSC_LR_Conv, [512, 3, 2, True, null, 0.125, 8]]
```

Sau scale YOLOv8n, số kênh thực tế là:

| Layer | Vị trí | Kênh thực tế | Rank | Conv thường | `DSC_LR_Conv` | Giảm |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 5 | Backbone P4/16 | `64 -> 128` | `8` | `73,984` | `2,512` | `96.60%` |
| 7 | Backbone P5/32 | `128 -> 256` | `16` | `295,424` | `8,096` | `97.26%` |
| 16 | Head P3 -> P4 | `64 -> 64` | `8` | `36,992` | `1,872` | `94.94%` |
| 19 | Head P4 -> P5 | `128 -> 128` | `16` | `147,712` | `5,792` | `96.08%` |

Ví dụ layer 5:

```text
Conv thường 64 -> 128:
3 * 3 * 64 * 128 + BN = 73,984 params

DSC_LR_Conv 64 -> 128:
rank = max(8, int(64 * 0.125)) = 8

Depthwise 3x3 = 3 * 3 * 64 + BN = 704
Reduce 1x1    = 64 * 8 + BN = 528
Expand 1x1    = 8 * 128 + BN = 1,280
Tổng          = 2,512 params
```

Kết luận: `DSC_LR_Conv` giúp các lớp Conv downsample giảm khoảng `95%` đến `97%` tham số.

## Phân tích cụ thể `LRTuckerC2f`

Hai lớp `LRTuckerC2f`:

```yaml
8:  [-1, 3, LRTuckerC2f, [1024, True, 1, 0.125, 0.125]]
21: [-1, 1, LRTuckerC2f, [1024, False, 1, 0.125, 0.125]]
```

Hiệu quả:

| Layer | Vị trí | Gốc | Custom | Giảm |
| --- | --- | ---: | ---: | ---: |
| 8 | Backbone P5 | `460,288` | `9,064` | `98.03%` |
| 21 | Head P5 | `493,056` | `10,088` | `97.95%` |

Lý do giảm mạnh:

- P5 là nơi số kênh cao nhất.
- `LRTuckerC2f` thay C2f bằng low-rank và Tucker decomposition.
- Bản mới dùng `e=0.125`, làm số kênh ẩn nhỏ hơn nhiều so với bản trước `e=0.5`.

## Phân tích cụ thể `LRFuseC2f`

Ba lớp `LRFuseC2f`:

```yaml
6:  [-1, 6, LRFuseC2f, [512, True, 1, 0.25, 0.5]]
12: [-1, 3, LRFuseC2f, [512, False, 1, 0.5, 0.5]]
18: [-1, 3, LRFuseC2f, [512, False, 1, 0.25, 0.5]]
```

Hiệu quả:

| Layer | Vị trí | Gốc | Custom | Giảm |
| --- | --- | ---: | ---: | ---: |
| 6 | Backbone P4 | `197,632` | `62,208` | `68.52%` |
| 12 | Head P5 -> P4 | `148,224` | `144,256` | `2.68%` |
| 18 | Head P4 | `123,648` | `42,080` | `65.97%` |

Layer 12 giảm ít vì cấu hình tại layer này vẫn giữ expansion `e=0.5` và phần input sau concat có số kênh lớn. `LRFuseC2f` chỉ thay phần fusion cuối, nên không phải lúc nào cũng giảm cực mạnh như `LRTuckerC2f`.

## Phân tích cụ thể `DSCDetect`

Detect head gốc:

```text
Detect: 751,507 params
```

Detect head custom:

```text
DSCDetect: 44,883 params
```

Mức giảm:

```text
751,507 - 44,883 = 706,624 params
Giảm khoảng 94.03%
```

Với input `640x640`, ba scale detect có kích thước:

```text
P3: 80x80
P4: 40x40
P5: 20x20
```

MACs ước tính trong detect head:

| Scale | Detect dùng Conv thường | `DSCDetect` | Giảm |
| --- | ---: | ---: | ---: |
| P3 `80x80` | `970.34M` | `67.58M` | `93.04%` |
| P4 `40x40` | `360.55M` | `20.38M` | `94.35%` |
| P5 `20x20` | `149.12M` | `6.84M` | `95.42%` |
| Tổng | `1480.01M` | `94.80M` | `93.59%` |

`DSCDetect` không thay đổi cách decode bbox cuối cùng. Nó chỉ làm nhẹ phần xử lý feature trong nhánh bbox và class.

## Kết luận

Các phương pháp optimize trong `yolov8n_custom_dsc_p45` có hiệu quả rõ nhất ở 3 nhóm:

1. `LRTuckerC2f` ở P5: giảm khoảng `98%` tham số tại layer 8 và 21.
2. `DSC_LR_Conv` ở các lớp downsample: giảm khoảng `95%` đến `97%` tham số.
3. `DSCDetect`: giảm khoảng `94%` tham số detect head và khoảng `93.59%` MACs detect head với input `640x640`.

Tổng thể, model giảm từ `3,011,043` params xuống `613,427` params, tương đương giảm `79.63%`. Đây là mức giảm lớn, nhưng vẫn cần đánh giá thực nghiệm bằng mAP, precision, recall và FPS để biết trade-off giữa tốc độ và độ chính xác.
