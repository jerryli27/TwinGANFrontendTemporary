// Constants
const kSketchKey = 83;
const kMagicWandMode = 0;
const kCropMode = 1;
const kSketchRefineMode = 2;
// Misc.

var mode = kMagicWandMode;

// Crop parameters
var cropPoints = [];

// Magic Wand Parameters
var colorThreshold = 15;
var blurRadius = 5;
var simplifyTolerant = 0;
var simplifyCount = 30;
var hatchLength = 4;
var hatchOffset = 0;

var imageInfo = null;
var canvasCache = [];
var cacheInd = null;
var mask = null;
var downPoint = null;
var allowDraw = false;
var currentThreshold = colorThreshold;

$(function () {
  var MAX_NUM_FACES = 4;
  image_id = "test_id";
  var debug_mode = true;
  let background = $("#background");
  let background_sketch = $("#background_sketch");
  let background_canvas = document.getElementById('background_canvas');


  // Initialization
  get_image();


  // Element properties.
  //$('#line_form').on('change', 'input[type="file"]', function(e) {
  $('#load_selfie_file').on('change', function (e) {
    var file = e.target.files[0],
      reader = new FileReader(),
      $preview = $(".preview");

    if (file.type.indexOf("image") < 0) {
      return false;
    }

    reader.onload = (function (file) {
      console.log("up");
      return function (e) {
        select_src(e.target.result);
      }
    })(file);

    reader.readAsDataURL(file);
  });

  // background.bind('load', function () {
  //   background_canvas.getContext('2d').drawImage(document.getElementById('background'), 0, 0);
  //   background_canvas.height = background.height();
  //   background_canvas.width = background.width();
  //   extractSketch();
  // });


  // // Add event listener for `click` events.
  // background_canvas.addEventListener('click', function (event) {
  //   let xy = {x: event.offsetX, y: event.offsetY};
  //   console.log(xy);
  //   let fillcolor = {a: 25, r: 255, g:0, b:0}
  //   let tolerance = 25;
  //   floodfill(xy.x,xy.y,fillcolor,background_canvas.getContext('2d'),background_canvas.width,background_canvas.height,tolerance);
  // }, false);

//--- functions 

  function uniqueid() {
    var idstr = String.fromCharCode(Math.floor((Math.random() * 25) + 65));
    do {
      var ascicode = Math.floor((Math.random() * 42) + 48);
      if (ascicode < 58 || ascicode > 64) {
        idstr += String.fromCharCode(ascicode);
      }
    } while (idstr.length < 32);
    return (idstr);
  }

  startExtractSketch = function () {
    $("#painting_label").show();
    $("#submit").prop("disabled", true);
    console.log("ExtractSketch started");
  };
  endExtractSketch = function () {
    $("#painting_label").hide();
    $("#submit").prop("disabled", false);
    console.log("ExtractSketch finished");
  };


  extractSketch = function () {
    startExtractSketch();
    if (debug_mode) {
      $('#output_panes').show();
      background_sketch.attr('src', background.attr("src"));
      background_sketch.height(background.height());
      background_sketch.width(background.width());
      return;
    }

    var ajaxData = new FormData();
    ajaxData.append('image', background.attr("src"));
    ajaxData.append('id', image_id);
    $.ajax({
      url: "/post",
      data: ajaxData,
      cache: false,
      contentType: false,
      processData: false,
      type: 'POST',
      dataType: 'json',
      success: function (data) {
        //location.reload();
        var now = new Date().getTime();
        $('#output_panes').show();

        $('#background_sketch').attr('src', '/static/images/extracted_sketches/' + image_id + '.png?' + now);
        endExtractSketch();
      },
      error: function (data, error_msg) {
        $('#output_panes').hide();

        alert("Got error message:" + error_msg.message + ". Please try again.");
        console.log(error_msg);
        endExtractSketch();
      }
    });
  };
});

// Global Key Press Functions
$(document).keydown(function (e) {
  if (e.which === kSketchKey) {
    // Sketch key pressed. Show sketch only.
    $("#background").hide();
  }
});
$(document).keyup(function (e) {
  if (e.which === kSketchKey) {
    // Sketch key up. Show actual image again.
    $("#background").show();
  }
});


// Magic Wand functions

function onRadiusChange(e) {
  blurRadius = e.target.value;
};

function initCanvas(img) {
  let cvs = document.getElementById("background_canvas");
  cvs.width = img.width;
  cvs.height = img.height;
  // let sketch_refine_cvs = document.getElementById("sketch_refine_canvas");
  // sketch_refine_cvs.width = img.width;
  // sketch_refine_cvs.height = img.height;
  imageInfo = {
    width: img.width,
    height: img.height,
    context: cvs.getContext("2d"),
  };
  mask = null;
  canvasCache = [];

  let temp_element = document.createElement("canvas");
  let tempCtx = temp_element.getContext("2d");
  tempCtx.canvas.width = imageInfo.width;
  tempCtx.canvas.height = imageInfo.height;
  tempCtx.drawImage(img, 0, 0);
  imageInfo.data = tempCtx.getImageData(0, 0, imageInfo.width, imageInfo.height);
};

function getMousePosition(e) {
  let p = $(e.target).offset(),
    x = Math.round((e.clientX || e.pageX) - p.left),
    y = Math.round((e.clientY || e.pageY) - p.top);
  return {x: x, y: y};
};

function onMouseDown(e) {
  // First backup current canvas.
  // Equivalent to this line: "res = imgData.data;"
  image_data_copy = new Uint8ClampedArray(imageInfo.context.getImageData(0, 0, imageInfo.width, imageInfo.height).data);
  canvasCache.push(image_data_copy);

  if (mode === kMagicWandMode) {
    if (e.button === 0) {
      allowDraw = true;
      downPoint = getMousePosition(e);
      drawMask(downPoint.x, downPoint.y);
    }
    else allowDraw = false;
  } else if (mode === kCropMode) {
    cropPoints = [getMousePosition(e)];
  } else if (mode === kSketchRefineMode) {
    console.log('sketch refine started.');
    let mouse_pos = getMousePosition(e);
    submit_sketch_refinement(mouse_pos);
  }else {
    alert('Unsupported mode.');
  }

};

function onMouseMove(e) {
  if (mode === kMagicWandMode) {
    if (allowDraw) {
      let p = getMousePosition(e);
      if (p.x != downPoint.x || p.y != downPoint.y) {
        let dx = p.x - downPoint.x,
          dy = p.y - downPoint.y,
          len = Math.sqrt(dx * dx + dy * dy),
          adx = Math.abs(dx),
          ady = Math.abs(dy),
          sign = adx > ady ? dx / adx : dy / ady;
        sign = sign < 0 ? sign / 5 : sign / 3;
        let thres = Math.min(Math.max(colorThreshold + Math.floor(sign * len), 1), 255);
        //let thres = Math.min(colorThreshold + Math.floor(len / 3), 255);
        if (thres != currentThreshold) {
          currentThreshold = thres;
          drawMask(downPoint.x, downPoint.y);
        }
      }
    }
  } else if (mode === kCropMode) {
    cropPoints.push(getMousePosition(e));
  }  else if (mode === kSketchRefineMode) {
    return;
  } else {
    alert('Unsupported mode.');
  }
};

function onMouseUp(e) {
  if (mode === kMagicWandMode) {
    allowDraw = false;
    currentThreshold = colorThreshold;
  } else if (mode === kCropMode) {
    cropPoints.push(getMousePosition(e));
    cropPoints = simplify(cropPoints);
    cropPoints.push(cropPoints[0]);

    let ctx = document.getElementById("background_canvas").getContext("2d");
    // Save the state, so we can undo the clipping
    ctx.save();
    // Create a shape, of some sort
    ctx.beginPath();
    ctx.moveTo(cropPoints[0].x, cropPoints[0].y);
    for (let i = 0; i < cropPoints.length; i++) {
      ctx.lineTo(cropPoints[i].x, cropPoints[i].y);
    }
    ctx.closePath();
    // Clip to the current path
    let color = 'rgba(25, 255, 255, ' + 25 / 255.0 + ')';
    ctx.strokeStyle = color;
    ctx.stroke();
    // ctx.rect(0, 0, imageInfo.width, imageInfo.height);

    // Naively calling fill() will result in incorrect transparency when two regions overlay each other.
    // Clear the area using compositing.
    ctx.globalCompositeOperation='destination-out';
    ctx.fillStyle='black';
    ctx.fill();
    ctx.globalCompositeOperation='source-over';

    ctx.fillStyle = color;
    ctx.fill();

    // Undo the clipping
    ctx.restore();
  } else if (mode === kSketchRefineMode) {
    return;
  } else {
    alert('Unsupported mode.');
  }
};

function showThreshold() {
  document.getElementById("threshold").innerHTML = "Threshold: " + currentThreshold;
};

function drawMask(x, y) {
  if (!imageInfo) return;

  showThreshold();

  let image = {
    data: imageInfo.data.data,
    width: imageInfo.width,
    height: imageInfo.height,
    bytes: 4
  };

  mask = MagicWand.floodFill(image, x, y, currentThreshold);
  mask = MagicWand.gaussBlurOnlyBorder(mask, blurRadius);
  // drawBorder();
  drawMaskOnCanvas();
};

function hatchTick() {
  hatchOffset = (hatchOffset + 1) % (hatchLength * 2);
  drawBorder(true);
};

function drawBorder(noBorder) {
  if (!mask) return;

  let x, y, i, j,
    w = imageInfo.width,
    h = imageInfo.height,
    ctx = imageInfo.context,
    imgData = ctx.createImageData(w, h),
    res = imgData.data;

  if (!noBorder) cacheInd = MagicWand.getBorderIndices(mask);

  ctx.clearRect(0, 0, w, h);

  let len = cacheInd.length;
  for (j = 0; j < len; j++) {
    i = cacheInd[j];
    x = i % w; // calc x by index
    y = (i - x) / w; // calc y by index
    k = (y * w + x) * 4;
    if ((x + y + hatchOffset) % (hatchLength * 2) < hatchLength) { // detect hatch color
      res[k + 3] = 255; // black, change only alpha
    } else {
      res[k] = 255; // white
      res[k + 1] = 255;
      res[k + 2] = 255;
      res[k + 3] = 255;
    }
  }

  ctx.putImageData(imgData, 0, 0);
};

function drawMaskOnCanvas() {
  if (!mask) return;
  let x, y, i, j,
    w = imageInfo.width,
    h = imageInfo.height,
    ctx = imageInfo.context,
    // imgData = ctx.createImageData(w, h),
    imgData = ctx.getImageData(0, 0, w, h),
    res = imgData.data;
  // Do not clear rectangle. Retain previous state.
  // ctx.clearRect(0, 0, w, h);

  let len = mask.data.length;
  for (i = 0; i < len; i++) {
    x = i % w; // calc x by index
    y = (i - x) / w; // calc y by index
    k = (y * w + x) * 4;
    if (mask.data[i] || (canvasCache[canvasCache.length - 1][k] != 0)) { // detect hatch color
      // rgba format.
      res[k] = 25; // light blue
      res[k + 1] = 255;
      res[k + 2] = 255;
      res[k + 3] = 25;
    } else {
      res[k] = 255; // white
      res[k + 1] = 255;
      res[k + 2] = 255;
      res[k + 3] = 0;
    }
  }
  ctx.putImageData(imgData, 0, 0);
};

function trace() {
  let cs = MagicWand.traceContours(mask);
  cs = MagicWand.simplifyContours(cs, simplifyTolerant, simplifyCount);

  mask = null;

  // draw contours
  let ctx = imageInfo.context;
  ctx.clearRect(0, 0, imageInfo.width, imageInfo.height);
  //inner
  ctx.beginPath();
  for (let i = 0; i < cs.length; i++) {
    if (!cs[i].inner) continue;
    let ps = cs[i].points;
    ctx.moveTo(ps[0].x, ps[0].y);
    for (let j = 1; j < ps.length; j++) {
      ctx.lineTo(ps[j].x, ps[j].y);
    }
  }
  ctx.strokeStyle = "red";
  ctx.stroke();
  //outer
  ctx.beginPath();
  for (let i = 0; i < cs.length; i++) {
    if (cs[i].inner) continue;
    let ps = cs[i].points;
    ctx.moveTo(ps[0].x, ps[0].y);
    for (let j = 1; j < ps.length; j++) {
      ctx.lineTo(ps[j].x, ps[j].y);
    }
  }
  ctx.strokeStyle = "blue";
  ctx.stroke();
};

function get_image() {
  $("#submit").prop("disabled", true);
  let ajaxData = new FormData();
  ajaxData.append('type', 'labeler');
  $.ajax({
    url: "/post",
    data: ajaxData,
    cache: false,
    contentType: false,
    processData: false,
    type: 'POST',
    dataType: 'json',
    success: post_success_callback,
    error: function (data, error_msg) {
      $('#output_panes').hide();
      alert("Got error message:" + error_msg.message + ". Please try again.");
      console.log(error_msg);
    }
  });
}

function clear_canvas(e) {
  imageInfo.context.clearRect(0, 0, imageInfo.width, imageInfo.height);
  console.log('clear finished.');
}

function save_annotation(skip=false) {
  $("#submit").prop("disabled", true);
  data = document.getElementById("background_canvas").toDataURL();

  let ajaxData = new FormData();
  ajaxData.append('type', 'labeler');
  if (!skip) {
    ajaxData.append('image', data);
  }
  ajaxData.append('id', image_id);
  ajaxData.append('skip', skip);
  $.ajax({
    url: "/post",
    data: ajaxData,
    cache: false,
    contentType: false,
    processData: false,
    type: 'POST',
    dataType: 'json',
    success: post_success_callback,
    error: function (data, error_msg) {
      $('#output_panes').hide();
      alert("Got error message:" + error_msg.message + ". Please try again.");
      console.log(error_msg);
    }
  });
}

function post_success_callback(data) {
  if (data.error) {
    alert("Got error message from successful server response:'" + data.error + "'.");
    return;
  }
  $("#img_pane").show(
    "fast", function () {
      let background = $("#background");
      let background_sketch = $("#background_sketch");

      background.attr("src", data.image); //  background.bind('load' ...) is called.

      background.bind('load', function () {
        background_sketch.attr('src', data.sketch);
        background_sketch.height(background.height());
        background_sketch.width(background.width());
        background_sketch.bind('load', function () {
          initCanvas(document.getElementById('background_sketch'));
          $("#submit").prop("disabled", false);
        });
      });
    });

  image_id = data.id_str;
}

function submit_sketch_refinement(mouse_pos) {
  console.log('submit_sketch_refinement initiated.');

  let temp_element = document.createElement("canvas");
  let tempCtx = temp_element.getContext("2d");
  tempCtx.canvas.width = imageInfo.width;
  tempCtx.canvas.height = imageInfo.height;

  // let sketch_refine_ctx = document.getElementById("sketch_refine_canvas").getContext("2d");
  let image_data = tempCtx.getImageData(mouse_pos.x, mouse_pos.y, imageInfo.width, imageInfo.height);
  image_data.data[3] = 255;
  tempCtx.putImageData(image_data, mouse_pos.x, mouse_pos.y);


  $("#submit").prop("disabled", true);
  data = temp_element.toDataURL();

  let ajaxData = new FormData();
  ajaxData.append('type', 'sketch_refinement');
  ajaxData.append('sketch', $("#background_sketch").attr('src'));
  ajaxData.append('sketch_refinement', data);
  ajaxData.append('id', image_id);
  ajaxData.append('x', mouse_pos.x);
  ajaxData.append('y', mouse_pos.y);
  $.ajax({
    url: "/post",
    data: ajaxData,
    cache: false,
    contentType: false,
    processData: false,
    type: 'POST',
    dataType: 'json',
    success: sketch_refinement_success_callback,
    error: function (data, error_msg) {
      $('#output_panes').hide();
      alert("Got error message:" + error_msg.message + ". Please try again.");
      console.log(error_msg);
    }
  });
}

function sketch_refinement_success_callback(data) {
  if (data.error) {
    alert("Got error message from successful server response:'" + data.error + "'.");
    return;
  }
  let background = $("#background");
  let background_sketch = $("#background_sketch");

  background_sketch.attr('src', data.refined_sketch);
  background_sketch.bind('load', function () {
    // let sketch_refine_ctx = document.getElementById("sketch_refine_canvas").getContext("2d");
    // sketch_refine_ctx.clearRect(0, 0, imageInfo.width, imageInfo.height);
    console.log('sketch_refinement_success_callback completed.');
  });

}

function undo(e) {
  if (!canvasCache.length) {
    alert("Cannot undo. No more stored actions.");
    return;
  }
  imageInfo.context.putImageData(new ImageData(canvasCache.pop(), imageInfo.width, imageInfo.height), 0, 0);
  console.log('undo finished.');
}

function change_mode(new_mode) {
  mode = new_mode;
  console.log(mode);
}