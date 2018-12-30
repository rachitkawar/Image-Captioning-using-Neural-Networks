<?php
   
	ini_set('max_execution_time', 300); 
   if(isset($_FILES['image'])){
     
      $errors= array();
      $file_name = $_FILES['image']['name'];
      $file_size =$_FILES['image']['size'];
      $file_tmp =$_FILES['image']['tmp_name'];
      $file_type=$_FILES['image']['type'];
      $file_ext=strtolower(end(explode('.',$_FILES['image']['name'])));
      
      $expensions= array("jpeg","jpg","png");
      
      if(in_array($file_ext,$expensions)=== false){
         $errors[]="extension not allowed, please choose a JPEG or PNG file.";
      }

     $run = "python CapGenerator/eval_model.py -i ".$file_name;
      if(empty($errors)==true){
      move_uploaded_file($file_tmp,$file_name);
		exec($run.' >err1.txt 2>&1');
      
      $line = '';
      $f = fopen("err1.txt","r");
      //$f = fopen('data.txt', 'r');
      $cursor = -1;
      
      fseek($f, $cursor, SEEK_END);
      $char = fgetc($f);

/**
* Trim trailing newline chars of the file
*/
      while ($char === "\n" || $char === "\r") {
         fseek($f, $cursor--, SEEK_END);
         $char = fgetc($f);
}

/**
* Read until the start of file or first newline char
*/
      while ($char !== false && $char !== "\n" && $char !== "\r") {
  /**
   * Prepend the new char
   */
      //$line = '';
      $line = $char.$line;
      fseek($f, $cursor--, SEEK_END);
      $char = fgetc($f);
}

#echo $line;
function bark($line)
      {
          
          //echo $line;
          echo "<script>const synth = window.speechSynthesis;</script>";
          echo "<script>var s = new SpeechSynthesisUtterance('$line');</script>";
          echo "<script>synth.speak(s);</script>";
      }
//bark($line);   
      
      }else{
         print_r($errors);
      }
   }
  
   
?>

<html>
<head>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
<script src="jquery.js"></script>  
<script type="text/javascript">
        
        function displayTimer(){
           localStorage.setItem("flag",true);
         document.getElementById("progress-bar").style.display = 'inline';
        }
        </script>
<link href="style.css" rel="stylesheet" type="text/css"/>
</head>
   <body title="Allow service to interact with desktop">
   <center>
   <h1 style="background:#225b84; color:#eff4f7">Image Captioning with voice </h1><br>
      <form action="" id="borderMe" method="POST" enctype="multipart/form-data">
         <input class="submitBtn" type="file" name="image" />
         <button class="submitBtn" onClick="displayTimer()" >submit</button> 
      </form>
      <div id="progress-bar" style="display: none;"> 
      <div>    
      <h2 id='staging' value=''>hello</h2>
    </div>
    <br/>
    <div class='progress-bar'>
       <canvas id='inactiveProgress' class='progress-inactive' height='275px' width='275px'></canvas>
      <canvas id='activeProgress' class='progress-active'  height='275px' width='275px'></canvas>
      <p>0%</p>
    </div>
  </div>
	  
  
   
 </div>

     
  <!-- progress bar -->
  

  
<!-- progress bar -->
   <?php
      if(isset($f)){
        echo "<img src='$file_name' style='height:510;width:640; align:center'>";
        echo "<p>$line</p>";
        bark($line); 
      } ?>
      </center>
   </body>  
</html>