$(function () {
    page_url = new URL(window.location.href);
    MAX_NUM_FACES = 4;
    IMAGE_HW = parseInt(page_url.searchParams.get("image_hw")) || 128;
    DO_WAIFU2X = (page_url.searchParams.get("do_waifu2x") === 'true') || false;
    image_id = "test_id";


    $("#painting_label").hide();
    $("#submit").prop("disabled", true);

    $("#submit").click(function () {
        if (!$("#background").attr("src")) {
            alert("select a file");
        } else {
            colorize();
        }
    });

    $('#img_pane').hide();


    //$('#line_form').on('change', 'input[type="file"]', function(e) {
    $('#load_selfie_file').on('change', function (e) {
        var file = e.target.files[0],
            reader = new FileReader(),
            $preview = $(".preview");

        if (file.type.indexOf("image") < 0) {
            return false;
        }

        reader.onload = (function (file) {
            console.log("up")
            return function (e) {
                select_src(e.target.result);
            }
        })(file);

        reader.readAsDataURL(file);
    });


    $("#set_line_url").click(function () {
        //currently not work
        select_src($("#load_line_url").val())
    });


    $('#face_cropped').bind('load', function () {
        $('#face_cropped')
//            .height(64)
//            .width(64)
    });

    $('#output').bind('load', function () {
        $('#output')
        //            .height( $('#background').height() )
        //            .width( $('#background').width() )
            .height(IMAGE_HW)
            .width(IMAGE_HW)
        $('#img_pane')
            .width($('#background').width() * 2.3 + 24)
            .height($('#background').height() + 20)
    });

    $(".download-combined").click(function (event) {
        if ($(event.target).is("a")) {
            console.log('clicked link');
            let image_subid = event.target.id.substr(event.target.id.length - 1);
            let ajaxData = new FormData();
            ajaxData.append('id', image_id);
            ajaxData.append('subid', image_subid);
            ajaxData.append('register_download', true);
            $.ajax({
                url: "/post",
                data: ajaxData,
                cache: false,
                contentType: false,
                processData: false,
                type: 'POST',
                dataType: 'json',
                success: function (data) {
                    console.log("Download registered.")
                },
                error: function (data, error_msg) {
                    console.log("Got error in registering download: " + error_msg);
                }

            });
        }
    });

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

    startPaint = function () {
        $("#painting_label").show();
        $("#submit").prop("disabled", true);
        console.log("domain transfer started");
    }
    endPaint = function () {
        $("#painting_label").hide();
        $("#submit").prop("disabled", false);
        console.log("domain transfer finished");
    }

    colorize = function () {
        startPaint()
        var ajaxData = new FormData();
        ajaxData.append('image', $("#background").attr("src"));
        ajaxData.append('id', image_id);
        ajaxData.append('do_waifu2x', DO_WAIFU2X);
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
                console.log("uploaded")
                var now = new Date().getTime();
                $('#output_panes').show();

                for (let i = 0; i < data.num_faces; i++) {
                    $('#output_pane_' + i).show();
                    // Breaks the cache to let the browser load the new image.
                    let output_dir = '/static/images/transferred_faces/';
                    if (DO_WAIFU2X) {
                        output_dir = '/static/images/transferred_faces_2x/';
                    }
                    $('#output_' + i).attr('src', output_dir + image_id + '_' + i + '.png?' + now);
                    $('#face_cropped_' + i).attr('src', '/static/images/cropped_faces/' + image_id + '_' + i + '.png?' + now);
                    $('#download_' + i).attr('href', '/static/images/combined/' + image_id + '_' + i + '.png?' + now);
                }
                for (let i = data.num_faces; i < MAX_NUM_FACES; i++) {
                    $('#output_pane_' + i).hide();
                }

                console.log(data.num_faces);
                endPaint()
            },
            error: function (data, error_msg) {
                $('#output_panes').hide();

                alert("Got error message:" + error_msg + ". Please try again.");
                console.log(error_msg);
                endPaint();
            }

        });
    }

    select_src = function (src) {//                complete:
        $("#img_pane").show(
            "fast", function () {
                image_id = uniqueid();

                $("#background").attr("src", src);
                $("#submit").prop("disabled", true);
                colorize();
            });
    };


});


