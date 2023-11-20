document.addEventListener("DOMContentLoaded", function() {
    const tocHeadings = document.querySelectorAll("#slothpy > div.toctree-wrapper");
    tocHeadings.forEach(heading => {
        const anchor = heading.querySelectorAll('li > ul > li > a');
        anchor.forEach(an => an.style.display="none");
    });
});

