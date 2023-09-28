$(document).ready(function() {
    // Hide class methods by default
    $('.wy-class-content').hide();

    // Add click event listener to class headers
    $('.wy-class-header').click(function() {
        // Toggle visibility of class methods when clicked
        $(this).next('.wy-class-content').slideToggle();
    });
});
