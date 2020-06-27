var template = '<div class="card">' +
    '<div class="card-header">' +
        '<a class="card-link" data-toggle="collapse" href="">' +
          '<div class="prediction"><b>Sentiment : </b><span class="badge"></span></div>' +
            '<div class="probability-score"><b>Prediction score : </b><span class="badge badge-info"></span></div>' +
        '</a>' +
    '</div>' +
    '<div id="collapse" class="collapse" data-parent="#accordion">' +
       '<div class="card-body">' +
         '<div class="news-title"><i><span class="title"></span></i></div>' +
         '<div class="news-body" style="border:1px solid grey"><span class="body"></span></div>' +
       '</div>' +
    '</div>' +
'</div>';
$(document).ready(function(){
var csrftoken = $("[name=csrfmiddlewaretoken]").val();
$('#btnSubmit').on('click', function(){
    $.ajax({
        beforeSend: function(xhr){
            $('div.note').css('display', 'block');
            $('div.error').css('display', 'none');
            $('div.news-wrapper').css('display', 'none');
            $('div.news-wrapper').empty();
            $('div.note h3').text('Your entity ' + $('#id_SearchText').val() + ' is being analysed...')
            xhr.setRequestHeader("X-CSRFToken", csrftoken);
        },
        type: "POST",
        url: "/process_articles/",
        data: { 'searchText' : $('#id_SearchText').val() },
        dataType: "json",
        success: function(result){
            console.log(result);
            $('div.note').css('display', 'none');
            if(!result.outcome){
                $('div.error').css('display', 'block');
                $('div.error').html('<h3>No information found for this entity</h3>');
            }
            else {
                  $('div.news-wrapper').css('display', 'block');
                  $.each(result.articlesList, function( idx, value ) {
                      var newsBlock = $($.parseHTML(template));
                      newsBlock.find('#collapse').attr('id','collapse'+idx);
                      newsBlock.find('.card-link').attr('href','#collapse'+idx);
                      newsBlock.find('.title').text(value[0]);
                      newsBlock.find('.body').html((value[1].length > 1700 ? value[1].substring(0,1700) : value[1]) + '...<i><a href="' + value[3] + '" target="_blank">Click here to know more</a></i>');
                      newsBlock.find('.badge-info').text(result.probabilityList[idx][1]);
                      var output = findSentiment(result.probabilityList[idx][1]);

                       newsBlock.find('.prediction .badge').addClass(output.split(',')[1]).text(output.split(',')[0]);
                      $('div.news-wrapper').append(newsBlock);
                  });
            }
        },
        error: function(jqXHR, status, error){
            $('div.note').css('display', 'none');
            $('div.error').css('display', 'block');
            $('div.error').html('<h3>Error : ' + error + '</h3>');
            //alert('Error: ' + jqXHR.responseText);
        }
    });
    return false;
    });
});

function findSentiment(prediction){
    if(prediction >= 0 && prediction <= 30){
        return 'Very Positive,badge-success'
    }
    else if(prediction > 30 && prediction <= 60){
        return 'Positive,badge-primary'
    }
    else if(prediction > 60 && prediction <= 75){
        return 'Neutral,badge-secondary'
    }
    else if(prediction > 75 && prediction <= 90){
        return 'Negative,badge-warning'
    }
    if(prediction > 90){
        return 'Very Negative,badge-danger'
    }
}